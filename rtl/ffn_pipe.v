// =====================================================================
// QECCT-Student  TransformerBlock FFN  (INT8, FPGA)  -- PIPELINED / speed-opt
//   Linear(32->128) -> GELU(LUT) -> Linear(128->32)     (per token)
//
// Same INT8 integer arithmetic as ffn.v (BIT-EXACT), pipelined for high Fmax.
//
// Architecture: a 32-lane MAC, time-shared (1 chunk/cycle).  The reduction
// dimension is processed in chunks of CH=32:
//   FC1 (reduce over 32)  -> 1 chunk  / neuron  (128 neurons)
//   FC2 (reduce over 128) -> 4 chunks / neuron  ( 32 neurons), accumulated
// Only 32 weights are read per cycle (vs 128 for a unified-wide MAC), which
// keeps the weight-ROM read-mux small -> fast synth + small area, mirroring
// the proven noise_estimator_pipe scale.
//
//   P0 register selected weights/operands (isolate ROM mux)
//   P1 32 parallel 8x8 products            (fabric mult; DSP saved for attention)
//   P2 adder tree 32 -> 8
//   P3 adder tree  8 -> 2
//   P4 chunk_sum = ps2[0]+ps2[1]           (products only, no bias yet)
//   P5 accumulate chunks: acc = (first?bias:acc)+chunk_sum; forward on last
//   P6 requant multiply  acc*M             (DSP)
//   P7 round-shift + clamp -> yq (INT8)
//   P8 FC1: GELU LUT -> hq ;  FC2: yq -> out_flat
//
// Drop-in interface-compatible with ffn.v  =>  same testbench / golden vectors.
// =====================================================================
`timescale 1ns/1ps
module ffn (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [255:0] in_flat,    // DM(32) x int8 input activation
    output reg          done,
    output reg  [255:0] out_flat    // DOUT(32) x int8 output activation
);
    `include "ffn_params.vh"
    localparam FC1 = 1'b0, FC2 = 1'b1;
    localparam integer CH        = 32;          // MAC lanes (chunk width)
    localparam integer NCHUNK2   = DFF/CH;      // FC2 chunks per neuron = 4
    localparam integer LAT       = 9;           // ISS -> writeback latency
    localparam integer FC2_START = 140;         // > 127 + LAT, so all hq[] ready
    localparam integer FC2_END   = FC2_START + DOUT*NCHUNK2;   // 140 + 128 = 268

    reg signed [7:0]  xin [0:DM-1];             // latched input vector
    reg signed [7:0]  hq  [0:DFF-1];            // FC1 results (GELU), FC2 inputs

    // ---------------- issue scheduler ----------------
    reg        running;
    reg [8:0]  mc;
    reg [5:0]  out_cnt;

    reg        iss_vld, iss_layer, iss_first, iss_last;
    reg [6:0]  iss_neuron;     // 0..127
    reg [1:0]  iss_chunk;      // 0..3
    reg [8:0]  r;
    always @* begin
        iss_vld=1'b0; iss_layer=FC1; iss_neuron=7'd0; iss_chunk=2'd0;
        iss_first=1'b1; iss_last=1'b1;
        if (running) begin
            if (mc < 9'd128) begin
                iss_vld=1'b1; iss_layer=FC1; iss_neuron=mc[6:0];
                iss_chunk=2'd0; iss_first=1'b1; iss_last=1'b1;
            end else if (mc >= FC2_START && mc < FC2_END) begin
                r = mc - FC2_START;
                iss_vld=1'b1; iss_layer=FC2;
                iss_neuron = r[8:2];           // r / 4
                iss_chunk  = r[1:0];           // r % 4
                iss_first  = (r[1:0]==2'd0);
                iss_last   = (r[1:0]==2'd3);
            end
        end
    end

    // ---------------- issue-stage operand/weight select (32 lanes) ----------------
    reg signed [7:0]  op_sel[0:CH-1];
    reg [255:0]       w_word_sel;     // one WROM word = 32 packed int8 weights
    reg [7:0]         w_addr;
    reg signed [31:0] bias_sel;
    reg signed [17:0] M_sel;
    integer s;
    always @* begin
        // single-port weight ROM read (no per-lane mux -> clean ROM inference)
        if (iss_layer==FC1) w_addr = iss_neuron;                          // 0..127
        else                w_addr = DFF[7:0] + iss_neuron*CHK + iss_chunk; // 128..255
        w_word_sel = WROM[w_addr];
        for (s=0; s<CH; s=s+1)
            op_sel[s] = (iss_layer==FC1) ? xin[s] : hq[iss_chunk*CH + s];
        bias_sel = (iss_layer==FC1) ? B1[iss_neuron] : B2[iss_neuron];
        M_sel    = (iss_layer==FC1) ? M1[17:0]       : M2[17:0];
    end

    // ---------------- pipeline registers ----------------
    reg [255:0]       w_word_r;                              // P0 (packed weights)
    reg signed [7:0]  op_r[0:CH-1];
    reg signed [31:0] bias_r;  reg signed [17:0] M_r;
    (* use_dsp = "no" *) reg signed [15:0] prod [0:CH-1];    // P1 (fabric mult)
    reg signed [31:0] ps1 [0:7];                             // P2 (32->8)
    reg signed [31:0] ps2 [0:1];                             // P3 ( 8->2)
    reg signed [31:0] csum_s4;                               // P4 chunk sum
    reg signed [31:0] acc_reg;                               // P5 running accumulator
    reg signed [31:0] acc_fwd_s5;                            // P5 forwarded acc (on last)
    (* use_dsp = "yes" *) reg signed [49:0] mul_s6;          // P6
    reg signed [7:0]  yq_s7;                                 // P7

    reg signed [31:0] bias_1, bias_2, bias_3, bias_4;        // bias -> P5
    reg signed [17:0] M_1, M_2, M_3, M_4, M_5;               // M    -> P6
    reg        v0,v1,v2,v3,v4,v5,v6,v7;
    reg        l0,l1,l2,l3,l4,l5,l6,l7;
    reg        f0,f1,f2,f3,f4;                               // first -> P5
    reg        e0,e1,e2,e3,e4,e5,e6,e7;                      // last  -> writeback valid
    reg [6:0]  o0,o1,o2,o3,o4,o5,o6,o7;                      // neuron idx

    integer i, g;
    reg signed [49:0] t, q;
    reg signed [31:0] acc_next;
    reg [8:0]         idx8;

    always @(posedge clk) begin
        if (rst) begin
            running<=0; mc<=0; out_cnt<=0; done<=0;
            v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0; out_flat<=0;
        end else begin
            done <= 1'b0;
            // -------- scheduler --------
            if (!running) begin
                if (start) begin
                    running<=1; mc<=0; out_cnt<=0;
                    for (i=0;i<DM;i=i+1) xin[i] <= in_flat[i*8 +: 8];
                end
            end else begin
                mc <= mc + 9'd1;
            end

            // -------- P0 : register the weight ROM word + operands (isolate ROM read) --------
            w_word_r <= w_word_sel;
            for (i=0;i<CH;i=i+1) op_r[i] <= op_sel[i];
            bias_r <= bias_sel; M_r <= M_sel;
            v0<=iss_vld; l0<=iss_layer; o0<=iss_neuron; f0<=iss_first; e0<=iss_last;

            // -------- P1 : 32 parallel products (unpack weights from the word) --------
            for (i=0;i<CH;i=i+1) prod[i] <= $signed(w_word_r[i*8 +: 8]) * op_r[i];
            bias_1<=bias_r; M_1<=M_r;
            v1<=v0; l1<=l0; o1<=o0; f1<=f0; e1<=e0;

            // -------- P2 : adder tree 32 -> 8 --------
            for (g=0;g<8;g=g+1)
                ps1[g] <= prod[4*g] + prod[4*g+1] + prod[4*g+2] + prod[4*g+3];
            bias_2<=bias_1; M_2<=M_1;
            v2<=v1; l2<=l1; o2<=o1; f2<=f1; e2<=e1;

            // -------- P3 : adder tree 8 -> 2 --------
            ps2[0] <= ps1[0] + ps1[1] + ps1[2] + ps1[3];
            ps2[1] <= ps1[4] + ps1[5] + ps1[6] + ps1[7];
            bias_3<=bias_2; M_3<=M_2;
            v3<=v2; l3<=l2; o3<=o2; f3<=f2; e3<=e2;

            // -------- P4 : chunk_sum (products only) --------
            csum_s4 <= ps2[0] + ps2[1];
            bias_4<=bias_3; M_4<=M_3;
            v4<=v3; l4<=l3; o4<=o3; f4<=f3; e4<=e3;

            // -------- P5 : accumulate chunks; forward acc on last --------
            acc_next = (f4 ? bias_4 : acc_reg) + csum_s4;
            if (v4) acc_reg <= acc_next;
            acc_fwd_s5 <= acc_next;
            M_5<=M_4;
            v5<=v4; l5<=l4; o5<=o4; e5<=e4;

            // -------- P6 : requant multiply (DSP) -- only meaningful on last chunk --
            mul_s6 <= acc_fwd_s5 * M_5;
            v6<=v5; l6<=l5; o6<=o5; e6<=e5;

            // -------- P7 : round-shift + clamp --------
            t = mul_s6 + (50'sd1 <<< (SSH-1));
            q = t >>> SSH;
            if      (q > 50'sd127)  yq_s7 <=  8'sd127;
            else if (q < -50'sd128) yq_s7 <= -8'sd128;
            else                    yq_s7 <= q[7:0];
            v7<=v6; l7<=l6; o7<=o6; e7<=e6;

            // -------- P8 : LUT (FC1) / direct (FC2) writeback (only on last chunk) --
            idx8 = $signed(yq_s7) + 9'sd128;
            if (v7 && e7 && l7==FC1) hq[o7] <= GELU_LUT[idx8[7:0]];
            if (v7 && e7 && l7==FC2) begin
                out_flat[o7*8 +: 8] <= yq_s7;
                out_cnt <= out_cnt + 6'd1;
                if (out_cnt == DOUT-1) begin done <= 1'b1; running <= 1'b0; end
            end
        end
    end
endmodule
