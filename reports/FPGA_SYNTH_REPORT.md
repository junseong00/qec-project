# noise_estimator — FPGA Implementation Report (Zynq-7020)

**Module:** `noise_estimator` (INT8) — FC(8→24) → GELU(LUT) → FC(24→9) → Sigmoid(LUT)
**Target:** AMD Xilinx Zynq-7020, `xc7z020clg400-1` (speed grade −1)
**Tool:** Vivado 2024.2, out-of-context (block-level) flow: `synth_design -mode out_of_context`
→ `opt_design` → `place_design` → `route_design`. Numbers below are **post-route**.
**Constraint:** single 100 MHz clock (10.000 ns).
**Functional status:** both versions verified **BIT-EXACT** vs the INT8 golden model in
xsim (20/20 vectors, 180/180 output bytes, 0 mismatches).

## Two design points (area-minimal vs speed-optimal)

| Metric | **Area-minimal** (`noise_estimator.v`) | **Speed-optimal** (`noise_estimator_pipe.v`) |
|--------|---------------------------------------:|---------------------------------------------:|
| Architecture | 1 neuron/cycle, single-cycle 24-MAC, **unpipelined** | 6-stage pipeline, parallel MAC + DSP, **pipelined** |
| Slice LUTs        | 3,310 (6.22 %) | **912 (1.71 %)** |
| Slice Registers (FF) | 293 (0.28 %) | 340 (0.32 %) |
| DSP48E1           | **0 (0 %)** | 26 (11.82 %) |
| Block RAM         | 0 (0 %) | 0 (0 %) |
| CARRY4            | 643 | 39 |
| **Fmax (routed)** | 17.9 MHz | **105.3 MHz** |
| Latency (cycles)  | 36 | 48 |
| Latency (time)    | ≈ 2.01 µs | **≈ 0.46 µs** |

*(Available on xc7z020: 53,200 LUT · 106,400 FF · 220 DSP · 140 BRAM.)*

### Reading the table
- **Speed-optimal is both faster AND smaller in LUTs.** Pipelining lets the 24-lane
  INT8 MAC and the requantization multiply map to **26 DSP48E1** (24 for the MAC adder-
  tree groups via the DSP cascade, 2 for the 18×18 requant multiply). That removes the
  large LUT/CARRY4 multiplier-accumulator fabric of the area-min version, so LUTs drop
  3,310 → 912 while Fmax rises 17.9 → 105.3 MHz (5.9×) and latency improves 4.4×.
- **Cost** is 26 DSP (12 % of the device) and 47 extra FF — a clear win on this part,
  which has DSP to spare.
- **No BRAM** in either: all weight/bias ROMs and the GELU/Sigmoid LUTs fit in
  distributed LUT-ROM (the model is only ~0.4 KB of INT8 params for this submodule).
- OOC (block-level) numbers: no IO buffers are counted — appropriate for an IP block
  driven from the PS over AXI in the full SoC.

## Numerical accuracy (INT8 vs float PyTorch)
Identical for both RTL versions (same golden arithmetic):
- 20 test vectors: **99.44 %** >0.5 binarization agreement, max |INT8−float| = 0.057.
- All 256 possible syndromes: 98.31 % agreement, max error 0.18.

## Files
- RTL: `rtl/noise_estimator.v` (area-min), `rtl/noise_estimator_pipe.v` (pipelined)
- Params/LUTs: `rtl/noise_estimator_params.vh`; testbench `rtl/tb_noise_estimator.v` (+ `rtl/tb_vectors.vh`)
- Raw reports: `reports/utilization_areamin_impl.txt`, `reports/utilization_pipe_impl.txt`,
  `reports/timing_pipe_impl.txt`

---

# FFN pipe FPGA Implementation Result (Zynq-7020)

**Module:** `ffn_pipe.v` (module name `ffn`) - TransformerBlock FFN, INT8  
**Architecture:** pipelined 32-lane time-multiplexed MAC, `Linear(32->128) -> GELU(LUT) -> Linear(128->32)`  
**Target:** AMD Xilinx Zynq-7020, `xc7z020clg400-1`  
**Tool/flow:** Vivado 2024.2 OOC `synth_design -> opt_design -> place_design -> route_design`  
**Constraint:** single 100 MHz clock (10.000 ns).  
**Functional status:** xsim **BIT-EXACT PASS** vs INT8 golden model, 100 vectors, 3200/3200 output bytes, 0 mismatches.

## FFN post-route result

| Metric | `ffn_pipe.v` post-route |
|--------|------------------------:|
| Slice LUTs | 4,477 (8.42 %) |
| Slice Registers (FF) | 2,905 (2.73 %) |
| DSP48E1 | 2 (0.91 %) |
| Block RAM | 0 (0.00 %) |
| CARRY4 | 484 |
| WNS @ 100 MHz | +1.842 ns |
| Timing status | MET |
| Estimated Fmax from WNS | ~122.6 MHz |
| Latency (cycles) | ~269 cycles |
| Latency @ 100 MHz | ~2.69 us |

### Notes
- This run synthesized only `rtl/ffn_pipe.v`; the single-cycle `rtl/ffn.v` baseline was not included.
- The packed `WROM` rewrite resolved the previous Vivado area-optimization stall and completed synth/P&R in batch mode.
- OOC timing has no top-level input/output delays and reports the usual OOC clock-source warnings, so these numbers are appropriate as block-level FPGA evidence, not final SoC sign-off timing.

## FFN files
- RTL: `rtl/ffn_pipe.v`
- Testbench/vectors: `rtl/tb_ffn.v`, `rtl/tb_ffn_vectors.vh`
- Raw reports: `reports/utilization_ffn_pipe_impl.txt`, `reports/timing_ffn_pipe_impl.txt`
