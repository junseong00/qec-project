// =====================================================================
// Testbench: drives NVEC input activation vectors through ffn,
// checks RTL output BIT-EXACT vs golden INTEGER reference (tb_ffn_vectors.vh).
// =====================================================================
`timescale 1ns/1ps
module tb_ffn;
    `include "tb_ffn_vectors.vh"
    localparam integer DM   = 32;
    localparam integer DOUT = 32;

    reg          clk = 0, rst = 1, start = 0;
    reg  [255:0] in_flat = 0;
    wire         done;
    wire [255:0] out_flat;

    ffn dut(.clk(clk), .rst(rst), .start(start),
            .in_flat(in_flat), .done(done), .out_flat(out_flat));

    always #5 clk = ~clk;   // 100 MHz

    integer v, i, k, errors, byte_mismatch;
    reg signed [7:0] got, exp;

    initial begin
        errors = 0; byte_mismatch = 0;
        @(negedge clk); rst = 0;
        @(negedge clk);

        for (v = 0; v < NVEC; v = v + 1) begin
            // pack input vector
            for (i = 0; i < DM; i = i + 1)
                in_flat[i*8 +: 8] = TB_IN[v*DM + i];
            start = 1; @(negedge clk); start = 0;
            while (!done) @(negedge clk);

            // ---- byte-exact compare (RTL vs golden integer) ----
            for (k = 0; k < DOUT; k = k + 1) begin
                got = out_flat[k*8 +: 8];
                exp = TB_OUT[v*DOUT + k];
                if (got !== exp) begin
                    byte_mismatch = byte_mismatch + 1;
                    errors = errors + 1;
                    $display("  [VEC %0d] out[%0d] MISMATCH: rtl=%0d golden=%0d", v, k, got, exp);
                end
            end
            @(negedge clk);
        end

        $display("");
        $display("==================================================");
        $display(" FFN BIT-EXACT CHECK (RTL vs INT8 golden integer model)");
        $display("   vectors          : %0d", NVEC);
        $display("   output bytes     : %0d", NVEC*DOUT);
        $display("   byte mismatches  : %0d", byte_mismatch);
        if (errors == 0)
            $display("   RESULT           : PASS  (RTL is BIT-EXACT)");
        else
            $display("   RESULT           : FAIL  (%0d errors)", errors);
        $display("==================================================");
        $finish;
    end

    // safety timeout
    initial begin #2000000; $display("TIMEOUT"); $finish; end
endmodule
