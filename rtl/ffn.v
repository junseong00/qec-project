// =====================================================================
// QECCT-Student  TransformerBlock FFN  (INT8, FPGA)
//   Linear(32->128) -> GELU(LUT) -> Linear(128->32)     (per token)
//
// Time-multiplexed: 1 neuron / cycle (FC1: 128 cyc, FC2: 32 cyc).
// All integer arithmetic mirrors gen_ffn.py golden model exactly
//   =>  RTL output is BIT-EXACT vs the INT8 golden reference.
//
//   requant(acc) = clamp( (M*acc + (1<<(SSH-1))) >>> SSH , -128, 127 )
//
// Unlike noise_estimator there is NO sigmoid: the output is a signed INT8
// activation (scale_out) that feeds the residual add in the full block.
//
// Handshake: pulse `start` with `in_flat`; `done` pulses 1 cyc when ready.
//   in_flat  : DM(32)   x 8-bit signed  (in[i]  = in_flat[i*8  +: 8])
//   out_flat : DOUT(32) x 8-bit signed  (out[k] = out_flat[k*8 +: 8])
// =====================================================================
`timescale 1ns/1ps
module ffn (
    input  wire         clk,
    input  wire         rst,        // synchronous, active high
    input  wire         start,
    input  wire [255:0] in_flat,    // DM(32) x int8 input activation
    output reg          done,
    output reg  [255:0] out_flat    // DOUT(32) x int8 output activation
);
    `include "ffn_params.vh"

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
    localparam S_IDLE=0, S_FC1=1, S_FC2=2, S_DONE=3;
    reg [1:0]         state;
    reg signed [7:0]  xin [0:DM-1];    // latched input vector
    reg signed [7:0]  hq  [0:DFF-1];   // GELU outputs (FC2 inputs)
    reg [7:0]         idx;             // neuron counter (0..127)

    integer i;
    reg signed [31:0] acc;
    reg signed [7:0]  yq;
    reg [8:0]         lut_idx;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; idx <= 0; out_flat <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            // ------------------------------------------------- IDLE
            S_IDLE: if (start) begin
                for (i=0;i<DM;i=i+1) xin[i] <= in_flat[i*8 +: 8];
                idx <= 0; state <= S_FC1;
            end
            // ------------------------------------------------- FC1 (1 neuron/cyc)
            S_FC1: begin
                acc = B1[idx];
                for (i=0;i<DM;i=i+1)
                    acc = acc + W1[idx*DM + i] * xin[i];
                yq      = requant(acc, M1);
                lut_idx = $signed(yq) + 9'sd128;
                hq[idx] <= GELU_LUT[lut_idx[7:0]];
                if (idx == DFF-1) begin idx <= 0; state <= S_FC2; end
                else idx <= idx + 1'b1;
            end
            // ------------------------------------------------- FC2 (1 neuron/cyc)
            S_FC2: begin
                acc = B2[idx];
                for (i=0;i<DFF;i=i+1)
                    acc = acc + W2[idx*DFF + i] * hq[i];
                yq = requant(acc, M2);
                out_flat[idx*8 +: 8] <= yq;
                if (idx == DOUT-1) begin idx <= 0; state <= S_DONE; end
                else idx <= idx + 1'b1;
            end
            // ------------------------------------------------- DONE
            S_DONE: begin done <= 1'b1; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
