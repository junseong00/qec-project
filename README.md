# ROQET QECCT FPGA Verification

## 소개
이 프로젝트는 양자 오류 정정(QEC)을 위한 QECCT 기반 온디바이스 AI 디코더의 핵심 연산 블록을 Verilog RTL로 구현하고, Vivado 기반 FPGA OOC 합성 및 검증 결과를 정리한 코드/자료 모음입니다.

범위는 전체 SoC를 완성하기보다, AI Engine 내부의 핵심 블록인 Noise Estimator와 Transformer FFN 블록이 RTL 수준에서 정확히 동작하고 FPGA에 구현 가능한지 확인하는 데 집중했습니다.

## 주요 기능
- QECCT Student 모델 기반 INT8 RTL 구현
- Noise Estimator 블록 RTL 구현 및 xsim bit-exact 검증
- Transformer FFN 블록(`ffn_pipe`) RTL 구현 및 xsim bit-exact 검증
- Vivado 2024.2 기반 Zynq-7020 OOC 합성 및 post-route 결과 분석
- LUT, FF, DSP, BRAM, WNS, Fmax, latency 등 FPGA 구현 지표 정리
- 발표 및 보고서용 블록 다이어그램, Vivado schematic, 합성 리포트 제공

## 사용 방법
1. 레포지토리를 다운로드합니다.
   ```bash
   git clone <repository-url>
   cd fpga_handoff
   ```

2. RTL 파라미터 및 테스트 벡터를 재생성합니다.
   ```bash
   python tools/gen_noise_estimator.py
   python tools/gen_ffn.py
   ```

3. Vivado xsim으로 RTL 기능 검증을 수행합니다.
   - Noise Estimator testbench: `rtl/tb_noise_estimator.v`
   - FFN testbench: `rtl/tb_ffn.v`

4. Vivado OOC 합성 및 구현 결과를 확인합니다.
   - 요약 리포트: `reports/FPGA_SYNTH_REPORT.md`
   - FFN 상태 문서: `reports/FFN_STATUS.md`
   - 원본 utilization/timing report: `reports/`

5. 발표/보고서용 그림 자료를 확인합니다.
   - 블록 다이어그램: `report_assets/figures/`
   - Vivado RTL schematic: `report_assets/vivado_schematics/`

## 프로젝트 구조
```text
fpga_handoff/
  rtl/             Verilog RTL, testbench, Vivado Tcl
  tools/           RTL parameter/test-vector generation scripts
  model/           QECCT Student model files and result JSON
  data/            FPGA test vectors
  golden/          INT8 golden model reports
  reports/         Vivado synthesis/timing/utilization reports
  report_assets/   발표 및 보고서용 그림 자료
  archive/         캡스톤 보고서, 발표자료, 원본 실험 자료
```

## 주요 결과
- Noise Estimator pipeline
  - xsim bit-exact PASS
  - LUT 912, FF 340, DSP 26, BRAM 0
  - Fmax 105.3 MHz
  - latency 약 0.46 us

- Transformer FFN (`ffn_pipe`)
  - xsim bit-exact PASS
  - LUT 4,477, FF 2,905, DSP 2, BRAM 0
  - WNS +1.842 ns @ 100 MHz
  - latency 약 2.69 us

