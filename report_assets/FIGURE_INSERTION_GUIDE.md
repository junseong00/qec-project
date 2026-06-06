# 최종보고서 그림 삽입 가이드

최종보고서 양식 기준으로 반드시 들어가면 좋은 그림과, 추가하면 보고서 완성도가 올라가는 그림을 정리했다.

## 양식상 들어가면 좋은 필수 그림

| 보고서 위치 | 권장 그림 | 이유 | 생성 파일 |
|---|---|---|---|
| 2장 주요 알고리즘 및 설계 방법 | 전체 SoC 구성도 | 양식에서 "아키텍처 또는 전체 시스템 구성도"를 요구함 | `report_assets/figures/system_architecture.svg` |
| 3장 최종 완료 결과 | FPGA 검증 절차도 | 실제로 FPGA 검증을 어떻게 했는지 설명하기 좋음 | `report_assets/figures/fpga_verification_flow.svg` |
| 3장 성능/테스트 결과 | FPGA post-route 결과 요약 | LUT/FF/DSP/BRAM/WNS 결과를 시각적으로 보여줌 | `report_assets/figures/fpga_results_summary.svg` |

## 추가하면 좋은 그림

| 보고서 위치 | 추가 권장 그림 | 설명 |
|---|---|---|
| 2장 알고리즘 설명 | QECCT Student 모델 구조도 | syndrome 입력부터 noise estimator, embedding, Transformer block, output_fc까지 흐름 표시 |
| 2장 Surface Code 설명 | distance-3 Surface Code 격자 그림 | stabilizer와 syndrome 개념을 비전공 심사위원에게 설명하기 좋음 |
| 3장 설계 모델 | RTL 모듈 계층도 | `noise_estimator`, `ffn_pipe`, testbench, parameter header 관계 표시 |
| 4장/5장 미완료 항목 | 캡스톤 I/II 로드맵 | 이번 학기 완료 범위와 다음 학기 확장 범위를 구분 |
| 발표자료 | Vivado 검증 단계 스크린샷 | 실제 `synth_design`, `route_design`, WNS 결과 캡처를 넣으면 신뢰도가 높아짐 |

## HWP에 넣을 때 추천 순서

1. `system_architecture.svg`
   - 2장 "하드웨어 설계 방법" 뒤에 삽입
   - 캡션: "그림 1. ROQET 온디바이스 QEC AI 가속기 SoC 구성"

2. `fpga_verification_flow.svg`
   - 2장 "검증 방법" 또는 3장 "FPGA 검증 결과" 앞에 삽입
   - 캡션: "그림 2. RTL 기능 검증 및 Vivado FPGA 구현 검증 절차"

3. `fpga_results_summary.svg`
   - 3장 "FPGA 검증 결과" 표 앞 또는 뒤에 삽입
   - 캡션: "그림 3. Vivado post-route 자원 및 타이밍 결과 요약"

## 주의할 표현

- "FPGA 보드 실측 결과"라고 쓰면 안 된다.
- "Vivado post-route 구현 검증 결과" 또는 "FPGA 디바이스 모델 기반 합성/P&R 결과"라고 쓰는 것이 정확하다.
- 전체 QECCT decoder가 완성된 것이 아니라, `noise_estimator`와 `ffn_pipe` 핵심 블록을 검증한 결과임을 명시해야 한다.
