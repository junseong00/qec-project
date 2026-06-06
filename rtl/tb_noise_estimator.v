// =====================================================================
// Testbench: drives 20 syndrome vectors through noise_estimator,
// checks RTL output BIT-EXACT vs golden INTEGER reference (tb_vectors.vh).
// =====================================================================
`timescale 1ns/1ps
module tb_noise_estimator;
    `include "tb_vectors.vh"
    localparam integer NOUT = 9;

    reg         clk = 0, rst = 1, start = 0;
    reg  [7:0]  syn = 0;
    wire        done;
    wire [71:0] out_flat;
    wire [8:0]  bin;

    noise_estimator dut(.clk(clk), .rst(rst), .start(start),
                        .syn(syn), .done(done), .out_flat(out_flat), .bin(bin));

    always #5 clk = ~clk;   // 100 MHz

    integer v, k, errors, byte_mismatch, bin_mismatch;
    reg [7:0] got, exp;

    initial begin
        errors = 0; byte_mismatch = 0; bin_mismatch = 0;
        @(negedge clk); rst = 0;
        @(negedge clk);

        for (v = 0; v < NVEC; v = v + 1) begin
            // launch
            syn = TB_SYN[v]; start = 1; @(negedge clk); start = 0;
            // wait for done
            while (!done) @(negedge clk);

            // ---- byte-exact compare (RTL vs golden integer) ----
            for (k = 0; k < NOUT; k = k + 1) begin
                got = out_flat[k*8 +: 8];
                exp = TB_OUT[v*NOUT + k];
                if (got !== exp) begin
                    byte_mismatch = byte_mismatch + 1;
                    errors = errors + 1;
                    $display("  [VEC %0d] out[%0d] MISMATCH: rtl=%0d golden=%0d", v, k, got, exp);
                end
            end
            // ---- binarization compare ----
            if (bin !== TB_BIN[v]) begin
                bin_mismatch = bin_mismatch + 1;
                errors = errors + 1;
                $display("  [VEC %0d] BIN MISMATCH: rtl=%b golden=%b", v, bin, TB_BIN[v]);
            end
            $display("VEC %02d  syn=%08b  rtl_out_bytes=%h  bin=%09b  %s",
                     v, TB_SYN[v], out_flat, bin, (errors==0||(byte_mismatch==0))?"":"");
            @(negedge clk);
        end

        $display("");
        $display("==================================================");
        $display(" BIT-EXACT CHECK (RTL vs INT8 golden integer model)");
        $display("   vectors          : %0d", NVEC);
        $display("   output bytes     : %0d", NVEC*NOUT);
        $display("   byte mismatches  : %0d", byte_mismatch);
        $display("   bin  mismatches  : %0d", bin_mismatch);
        if (errors == 0)
            $display("   RESULT           : PASS  (RTL is BIT-EXACT)");
        else
            $display("   RESULT           : FAIL  (%0d errors)", errors);
        $display("==================================================");
        $finish;
    end

    // safety timeout
    initial begin #100000; $display("TIMEOUT"); $finish; end
endmodule
