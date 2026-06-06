# FFN block — status & resume guide (read THIS file to continue)

**Date:** 2026-06-02  **Block:** TransformerBlock FFN `Linear(32→128) → GELU → Linear(128→32)`,
INT8, target Zynq-7020 (`xc7z020clg400-1`), OOC flow, 100 MHz goal.
Parent context: `HANDOFF_FPGA.md` + memory `qecct-fpga-verification`. This file is self-contained
enough to resume the FFN work.

---

## Update: synthesis/P&R completed on 2026-06-04

`rtl/synth_ffn_pipe.tcl` completed Vivado 2024.2 OOC synth/place/route for `rtl/ffn_pipe.v`
on Zynq-7020 (`xc7z020clg400-1`). The single-cycle `rtl/ffn.v` was not synthesized.

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

Raw reports:
- `reports/utilization_ffn_pipe_impl.txt`
- `reports/timing_ffn_pipe_impl.txt`

The packed `WROM` mitigation worked; the previous area-optimization stall did not recur.

---

## ① Functional: `ffn_pipe.v` is BIT-EXACT ✅
- **`rtl/ffn_pipe.v`** (module name `ffn`) — pipelined "speed-opt" FFN.
  - 32-lane MAC, time-multiplexed (1 chunk/cycle). FC1 reduces over 32 in 1 chunk;
    FC2 reduces over 128 in **4 chunks of 32, accumulated** across cycles. 9-stage pipeline:
    products in LUT fabric, requant multiply in DSP. Weights are read as a **single 256-bit
    packed `WROM` word per cycle** (32 int8 packed) and unpacked after a register stage.
  - **xsim result: PASS — 100 vectors, 3200/3200 output bytes, 0 mismatches** vs the INT8
    golden-integer model. Re-verify any time:
    ```
    cd rtl && rm -rf s && mkdir s && cd s
    "/c/Xilinx/Vivado/2024.2/bin/xvlog.bat" --nolog -i .. ../ffn_pipe.v ../tb_ffn.v
    "/c/Xilinx/Vivado/2024.2/bin/xelab.bat" --nolog -i .. work.tb_ffn -s s
    "/c/Xilinx/Vivado/2024.2/bin/xsim.bat"  --nolog s -R
    ```
- **`rtl/ffn.v`** — area-min single-cycle baseline, also BIT-EXACT, but its 128-term
  combinational MAC **explodes synth → do NOT synthesize it.** Keep as documented baseline.
- INT8-vs-float accuracy (`golden/golden_ffn_report.txt`, 4692 tokens): **max 0.047, rel-RMS 3.39%,
  sign-agreement 98.23%.**
- Regenerate everything: `python tools/gen_ffn.py` (block 0; `FFN_BLOCK=1` for block 1). Emits
  `rtl/ffn_params.vh` (flat arrays + packed `WROM[0:255]`), `rtl/tb_ffn_vectors.vh`,
  `golden/golden_ffn_report.txt`.

## ② Synthesis: NOT yet completed — no util/timing numbers yet ⚠️
Every OOC synth attempt stalled/failed; **we do not yet have LUT/FF/DSP/Fmax for the FFN.**
Two distinct blockers were hit:
1. **Win11 `NoDefaultCurrentDirectoryInExePath` policy breaks `synth_design`'s multithread
   helper child** (`couldn't read .../unimacro_vhdl.tcl: No error` right after
   "Launching helper process for spawning children"). This silently killed both the batch and
   the in-memory MCP runs (the stray extra `vivado.exe` processes were dead helper children).
   *Mitigation applied:* `set_param general.maxThreads 1` (already in `rtl/synth_ffn_pipe.tcl`)
   **+** `export NoDefaultCurrentDirectoryInExePath=0` before launching vivado. This got synth
   *past* the helper step.
2. **Area-optimization hang on wide weight-ROM reads.** The pre-packed RTL read 32 weight lanes
   from one ROM (`W[neuron*W+s]`), which Vivado treated as a 32-read-port memory and spun
   **>38 min of CPU in "Start Cross Boundary and Area Optimization"** with no progress, then was
   killed. *Mitigation applied (but NOT yet validated by a completed synth):* the **packed
   single-port `WROM`** rewrite above — only proven in xsim so far, never run through a full
   `synth_design` to completion. **First thing next session: try synth with the packed ROM and
   see if the area-opt hang is gone.**

Permanent root-cause cure for blocker #1 (do once, persists): 
`reg add "HKCU\Environment" /v NoDefaultCurrentDirectoryInExePath /d 0 /f` (re-login to take effect).

## ③ Next synthesis strategies (in order to try)
1. **Re-run with the packed `WROM` fix** (current `ffn_pipe.v`) via the batch flow below and
   watch `synth.log`. If it now finishes in a few minutes → capture util/timing, done.
2. If area-opt still slow, add a runtime-biased directive: `synth_design ... -directive RuntimeOptimized`
   (or `-flatten_hierarchy none`), and/or `-no_lc`. Also try `opt_design -directive ExploreArea` off.
3. **Synthesize FC1 and FC2 as separate sub-modules / separate OOC runs** (split the design) so
   each weight ROM and MAC is smaller and optimized independently, then report combined numbers.
   The packed-ROM idea applies per-sub-module.
4. **Fall back to a stronger Linux machine** (the original lab server) for synth+P&R — this
   Galaxybook (16 GB RAM, Win11 policy quirks) is the bottleneck. RTL is portable; just copy
   `rtl/ffn_pipe.v`, `rtl/ffn_params.vh`, and `rtl/synth_ffn_pipe.tcl`. On Linux the multithread
   helper policy issue disappears entirely.

### Batch synth command (observable, single-threaded)
```
cd rtl && mkdir ffn_synthN && cd ffn_synthN          # fresh dir each time (old dirs lock)
export NoDefaultCurrentDirectoryInExePath=0
"/c/Xilinx/Vivado/2024.2/bin/vivado.bat" -mode batch -source ../synth_ffn_pipe.tcl \
    -log synth.log -journal synth.jou > console.out 2>&1
# watch:  tail -f synth.log   (markers: "Starting Synthesize", "Start Cross Boundary",
#          "Starting Placer", "Starting Routing", "FFN_PIPE_DONE")
```
`rtl/synth_ffn_pipe.tcl` runs synth→opt→place→route and writes
`reports/utilization_ffn_pipe_impl.txt` + `reports/timing_ffn_pipe_impl.txt`.
After a clean run: fill the FFN row into `reports/FPGA_SYNTH_REPORT.md` and update memory.

**Hygiene between attempts:** `taskkill //F //IM vivado.exe` then `rm -rf .Xil`. In-memory
synth via vivado-mcp blocks the tcl interpreter (run_tcl just times out) — prefer batch runs.

## Files
- RTL: `rtl/ffn_pipe.v` (deliverable), `rtl/ffn.v` (baseline, don't synth), `rtl/tb_ffn.v`
- Gen/params: `tools/gen_ffn.py`, `rtl/ffn_params.vh`, `rtl/tb_ffn_vectors.vh`, `golden/golden_ffn_report.txt`
- Synth: `rtl/synth_ffn_pipe.tcl`
