// =====================================================================
// QECCT-Student  noise_estimator  (INT8, FPGA)
//   Linear(8->24) -> GELU(LUT) -> Linear(24->9) -> Sigmoid(LUT)
//
// Time-multiplexed: 1 neuron / cycle (FC1: 24 cyc, FC2: 9 cyc).
// All integer arithmetic mirrors gen_noise_estimator.py golden model
// exactly  =>  RTL output is BIT-EXACT vs the INT8 golden reference.
//
//   requant(acc) = clamp( (M*acc + (1<<(SSH-1))) >>> SSH , -128, 127 )
//
// Handshake: pulse `start` with `syn`; `done` pulses 1 cyc when result ready.
//   out_flat : NOUT bytes, uint8 = round(sigmoid*255)  (out[k]=out_flat[k*8 +: 8])
//   bin      : NOUT bits, bit k = (sigmoid(y2[k])>0.5) = (out_q[k] >= 128)
// =====================================================================
`timescale 1ns/1ps
module noise_estimator (
    input  wire        clk,
    input  wire        rst,        // synchronous, active high
    input  wire        start,
    input  wire [7:0]  syn,        // bit i = syndrome[i]  (8 bits)
    output reg         done,
    output reg  [71:0] out_flat,   // NOUT(9) x 8-bit uint8
    output reg  [8:0]  bin         // >0.5 binarization
);
    `include "noise_estimator_params.vh"

    // ------- requant (round-half-up, arithmetic shift = floor) -------
    function signed [7:0] requant;
        input signed [31:0] acc;
        input signed [31:0] M;
        reg   signed [63:0] t;
        begin
            t = ($signed(acc) * $signed(M)) + (64'sd1 <<< (SSH-1));
            t = t >>> SSH;
            if (t > 64'sd127)        requant =  8'sd127;
            else if (t < -64'sd128)  requant = -8'sd128;
            else                     requant = t[7:0];
        end
    endfunction

    // ------- state -------
    localparam S_IDLE=0, S_FC1=1, S_GELU=2, S_FC2=3, S_DONE=4;
    reg [2:0]  state;
    reg [7:0]  syn_r;
    reg [5:0]  idx;                 // neuron counter (0..23)
    reg signed [7:0] hq [0:HID-1];  // GELU outputs (FC2 inputs)

    integer i;
    reg signed [31:0] acc;
    reg signed [7:0]  yq;
    reg [8:0]         lut_idx;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; idx <= 0; bin <= 0; out_flat <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            // ------------------------------------------------- IDLE
            S_IDLE: if (start) begin
                syn_r <= syn; idx <= 0; state <= S_FC1;
            end
            // ------------------------------------------------- FC1 (1 neuron/cyc)
            S_FC1: begin
                acc = B1[idx];
                for (i=0;i<NIN;i=i+1)
                    if (syn_r[i]) acc = acc + W1[idx*NIN + i];  // x in {0,1}
                yq      = requant(acc, M1);
                lut_idx = $signed(yq) + 9'sd128;
                hq[idx] <= GELU_LUT[lut_idx[7:0]];
                if (idx == HID-1) begin idx <= 0; state <= S_FC2; end
                else idx <= idx + 1'b1;
            end
            // ------------------------------------------------- FC2 (1 neuron/cyc)
            S_FC2: begin
                acc = B2[idx];
                for (i=0;i<HID;i=i+1)
                    acc = acc + W2[idx*HID + i] * hq[i];
                yq      = requant(acc, M2);
                lut_idx = $signed(yq) + 9'sd128;
                out_flat[idx*8 +: 8] <= SIG_LUT[lut_idx[7:0]];
                bin[idx]             <= SIG_LUT[lut_idx[7:0]][7]; // >=128
                if (idx == NOUT-1) begin idx <= 0; state <= S_DONE; end
                else idx <= idx + 1'b1;
            end
            // ------------------------------------------------- DONE
            S_DONE: begin done <= 1'b1; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
