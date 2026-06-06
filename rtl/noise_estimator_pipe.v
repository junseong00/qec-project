// =====================================================================
// QECCT-Student  noise_estimator  (INT8, FPGA)  -- PIPELINED / speed-opt
//   Linear(8->24) -> GELU(LUT) -> Linear(24->9) -> Sigmoid(LUT)
//
// Same INT8 integer arithmetic as noise_estimator.v (BIT-EXACT), pipelined
// for high Fmax. Per-neuron datapath (1 neuron/cycle throughput):
//
//   P1 24x parallel 8x8 products (DSP, registered)   <-- no serial cascade
//   P2 adder tree level 1   (24 -> 6 partial sums)
//   P3 adder tree level 2 + bias  -> acc
//   P4 requant multiply  acc*M     (DSP)
//   P5 round-shift + clamp -> yq (INT8)
//   P6 GELU/Sigmoid LUT + writeback
//
// One unified 24-lane MAC is time-shared by FC1 (8 active lanes, binary
// operands) and FC2 (24 lanes, INT8 operands).  Drop-in interface-compatible
// with noise_estimator.v  =>  same testbench / golden vectors.
// =====================================================================
`timescale 1ns/1ps
module noise_estimator (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [7:0]  syn,
    output reg         done,
    output reg  [71:0] out_flat,
    output reg  [8:0]  bin
);
    `include "noise_estimator_params.vh"
    localparam FC1 = 1'b0, FC2 = 1'b1;

    reg signed [7:0]  hq [0:HID-1];          // FC1 results (GELU), FC2 inputs

    // ---------------- issue scheduler ----------------
    //   mc 0..23  : FC1 neuron 0..23
    //   mc 24..33 : bubbles (drain FC1; LAT=7 < gap so all hq[] written)
    //   mc 34..42 : FC2 neuron 0..8
    //   done when 9th FC2 output is written back
    reg        running;
    reg [7:0]  syn_r;
    reg [7:0]  mc;
    reg [3:0]  out_cnt;

    reg        iss_vld, iss_layer;
    reg [4:0]  iss_idx;
    always @* begin
        iss_vld = 1'b0; iss_layer = FC1; iss_idx = 5'd0;
        if (running) begin
            if (mc < 8'd24)                     begin iss_vld=1'b1; iss_layer=FC1; iss_idx=mc[4:0];        end
            else if (mc >= 8'd34 && mc < 8'd43) begin iss_vld=1'b1; iss_layer=FC2; iss_idx=mc[4:0]-5'd34; end
        end
    end

    // ---------------- issue-stage operand/weight select ----------------
    reg signed [7:0]  w_sel [0:HID-1];
    reg signed [7:0]  op_sel[0:HID-1];
    reg signed [31:0] bias_sel;
    reg signed [17:0] M_sel;
    integer s;
    always @* begin
        for (s=0; s<HID; s=s+1) begin
            if (iss_layer==FC1) begin
                w_sel[s]  = (s<NIN) ? W1[iss_idx*NIN + s] : 8'sd0;
                op_sel[s] = (s<NIN) ? (syn_r[s] ? 8'sd1 : 8'sd0) : 8'sd0;
            end else begin
                w_sel[s]  = W2[iss_idx*HID + s];
                op_sel[s] = hq[s];
            end
        end
        bias_sel = (iss_layer==FC1) ? B1[iss_idx] : B2[iss_idx];
        M_sel    = (iss_layer==FC1) ? M1[17:0]    : M2[17:0];
    end

    // ---------------- pipeline registers ----------------
    reg signed [7:0]  w_r [0:HID-1];                         // P0 (isolates ROM mux)
    reg signed [7:0]  op_r[0:HID-1];
    reg signed [31:0] bias_r;  reg signed [17:0] M_r;
    reg        v0;  reg l0;  reg [4:0] o0;
    (* use_dsp = "yes" *) reg signed [15:0] prod [0:HID-1];   // P1
    reg signed [17:0] ps1 [0:5];                              // P2 partial sums
    reg signed [31:0] acc_s3;                                 // P3
    (* use_dsp = "yes" *) reg signed [49:0] mul_s4;           // P4
    reg signed [7:0]  yq_s5;                                  // P5

    reg signed [31:0] bias_s1, bias_s2;
    reg signed [17:0] M_s1, M_s2, M_s3;
    reg        v1,v2,v3,v4,v5;  reg l1,l2,l3,l4,l5;  reg [4:0] o1,o2,o3,o4,o5;

    integer i, g;
    reg signed [49:0] t, q;
    reg [7:0]         res6;
    reg [8:0]         idx6;

    always @(posedge clk) begin
        if (rst) begin
            running<=0; mc<=0; out_cnt<=0; done<=0;
            v0<=0; v1<=0; v2<=0; v3<=0; v4<=0; v5<=0; out_flat<=0; bin<=0;
        end else begin
            done <= 1'b0;
            // -------- scheduler --------
            if (!running) begin
                if (start) begin running<=1; syn_r<=syn; mc<=0; out_cnt<=0; end
            end else begin
                mc <= mc + 8'd1;
            end

            // -------- P0 : register selected weights/operands (isolate ROM mux) --------
            for (i=0;i<HID;i=i+1) begin w_r[i] <= w_sel[i]; op_r[i] <= op_sel[i]; end
            bias_r <= bias_sel; M_r <= M_sel;
            v0 <= iss_vld; l0 <= iss_layer; o0 <= iss_idx;

            // -------- P1 : 24 parallel products --------
            for (i=0;i<HID;i=i+1) prod[i] <= w_r[i] * op_r[i];
            bias_s1 <= bias_r; M_s1 <= M_r;
            v1 <= v0; l1 <= l0; o1 <= o0;

            // -------- P2 : adder tree level 1 (24 -> 6) --------
            for (g=0;g<6;g=g+1)
                ps1[g] <= prod[4*g] + prod[4*g+1] + prod[4*g+2] + prod[4*g+3];
            bias_s2 <= bias_s1; M_s2 <= M_s1;
            v2 <= v1; l2 <= l1; o2 <= o1;

            // -------- P3 : adder tree level 2 + bias -> acc --------
            acc_s3 <= bias_s2 + ps1[0] + ps1[1] + ps1[2] + ps1[3] + ps1[4] + ps1[5];
            M_s3 <= M_s2;
            v3 <= v2; l3 <= l2; o3 <= o2;

            // -------- P4 : requant multiply (DSP) --------
            mul_s4 <= acc_s3 * M_s3;
            v4 <= v3; l4 <= l3; o4 <= o3;

            // -------- P5 : round-shift + clamp --------
            t = mul_s4 + (50'sd1 <<< (SSH-1));
            q = t >>> SSH;
            if      (q > 50'sd127)  yq_s5 <=  8'sd127;
            else if (q < -50'sd128) yq_s5 <= -8'sd128;
            else                    yq_s5 <= q[7:0];
            v5 <= v4; l5 <= l4; o5 <= o4;

            // -------- P6 : LUT lookup + writeback --------
            idx6 = $signed(yq_s5) + 9'sd128;
            if (v5 && l5==FC1) hq[o5] <= GELU_LUT[idx6[7:0]];
            if (v5 && l5==FC2) begin
                res6 = SIG_LUT[idx6[7:0]];
                out_flat[o5*8 +: 8] <= res6;
                bin[o5]             <= res6[7];        // >=128  <=>  sigmoid>0.5
                out_cnt <= out_cnt + 4'd1;
                if (out_cnt == NOUT-1) begin done <= 1'b1; running <= 1'b0; end
            end
        end
    end
endmodule
