# Vivado 기반 FPGA 검증 설명 정리

이 문서는 최종보고서와 발표에서 박준성이 설명할 Vivado/FPGA 검증 파트를 정리한 것이다.

## 1. 이번 FPGA 검증에서 실제로 한 일

이번 프로젝트에서 수행한 FPGA 검증은 실물 FPGA 보드에 bitstream을 다운로드하여 측정한 실측 검증은 아니다. 대신 Vivado가 제공하는 FPGA 디바이스 모델을 사용하여 Verilog RTL이 특정 FPGA 칩에 실제로 합성, 배치, 배선 가능한지 확인한 post-route 구현 검증이다.

검증 대상 FPGA 디바이스는 다음과 같다.

| 항목 | 내용 |
|---|---|
| FPGA 계열 | AMD Xilinx Zynq-7020 |
| Part name | `xc7z020clg400-1` |
| Tool | Vivado 2024.2 |
| 검증 방식 | OOC, out-of-context, 블록 단위 합성/P&R |
| Clock constraint | 100 MHz, 10 ns |

검증한 RTL 블록은 다음 두 종류이다.

| 블록 | 파일 | 설명 |
|---|---|---|
| noise estimator | `rtl/noise_estimator.v`, `rtl/noise_estimator_pipe.v` | syndrome 입력으로부터 noise feature를 추정하는 초기 FC 블록 |
| FFN pipe | `rtl/ffn_pipe.v` | Transformer block 내부 FFN, `Linear(32->128) -> GELU -> Linear(128->32)` |

전체 QECCT decoder top-level을 FPGA에 모두 통합한 것은 아니며, 캡스톤 I에서는 핵심 연산 블록을 우선 검증하였다.

## 2. 검증 흐름

전체 흐름은 다음과 같다.

```text
PyTorch QECCT Student 모델
        |
        v
INT8 golden integer model 생성
        |
        v
Verilog RTL 구현
        |
        v
xsim 기능 검증
        |
        v
Vivado synthesis
        |
        v
Vivado place & route
        |
        v
utilization / timing report 확인
```

각 단계의 의미는 다음과 같다.

| 단계 | 의미 |
|---|---|
| INT8 golden model | PyTorch 모델을 FPGA 구현에 맞는 정수 연산 기준 모델로 변환 |
| Verilog RTL | 같은 연산을 FPGA에서 구현 가능한 하드웨어 로직으로 작성 |
| xsim | RTL 출력과 INT8 golden 출력이 완전히 일치하는지 기능 검증 |
| synthesis | RTL을 LUT, FF, DSP, BRAM 등 FPGA 논리 자원으로 변환 |
| place | 변환된 회로를 FPGA 내부 물리 위치에 배치 |
| route | 배치된 회로 사이의 배선을 FPGA routing 자원으로 연결 |
| timing analysis | 100 MHz clock에서 신호가 시간 안에 도착하는지 분석 |

## 3. 보드 없이 Vivado 결과가 나오는 이유

Vivado는 `xc7z020clg400-1` FPGA의 내부 구조 정보를 알고 있다. 즉, 해당 칩에 LUT, FF, DSP, BRAM이 몇 개 있는지, 각 자원이 어디에 배치되어 있는지, 배선 자원이 어떻게 구성되어 있는지, 각 경로의 지연시간이 얼마인지에 대한 디바이스 데이터베이스를 가지고 있다.

따라서 실물 보드를 연결하지 않아도 Vivado는 다음을 계산할 수 있다.

- RTL을 구현하는 데 필요한 LUT/FF/DSP/BRAM 수
- 회로가 FPGA 내부에 배치 가능한지 여부
- 배선이 가능한지 여부
- 100 MHz timing constraint를 만족하는지 여부
- WNS, TNS, Fmax 추정값

다만 이것은 실제 보드 전력, 온도, 외부 I/O, 물리적 측정 latency까지 검증한 것은 아니다. 따라서 표현은 다음처럼 해야 한다.

좋은 표현:

> Vivado post-route 구현 검증을 통해 Zynq-7020 디바이스 모델 기준의 자원 사용량과 timing closure를 확인하였다.

피해야 할 표현:

> 실물 FPGA 보드에서 동작을 실측하였다.

## 4. 기능 검증 결과

기능 검증은 xsim에서 수행하였다. 입력 벡터를 RTL과 INT8 golden model에 동시에 넣고, 출력이 byte 단위로 일치하는지 비교하였다.

| 블록 | 기능 검증 결과 |
|---|---|
| noise estimator | 20개 vector, 180/180 output byte 일치, mismatch 0 |
| FFN pipe | 100개 vector, 3200/3200 output byte 일치, mismatch 0 |

이 결과는 RTL이 의도한 정수 연산 모델과 동일하게 동작한다는 의미이다. 즉, “대략 비슷하다”가 아니라 bit-exact하게 일치했다.

## 5. Vivado post-route 결과

### 5.1 noise estimator

`noise_estimator`는 두 가지 구조로 비교하였다.

| 항목 | Area-minimal | Speed-optimal pipeline |
|---|---:|---:|
| Slice LUT | 3,310, 6.22% | 912, 1.71% |
| Slice FF | 293, 0.28% | 340, 0.32% |
| DSP48E1 | 0 | 26, 11.82% |
| BRAM | 0 | 0 |
| CARRY4 | 643 | 39 |
| Fmax | 17.9 MHz | 105.3 MHz |
| Latency | 약 2.01 us | 약 0.46 us |

해석:

- area-minimal은 DSP를 거의 쓰지 않고 LUT 중심으로 구현한 버전이다.
- speed-optimal은 pipeline과 DSP48E1을 사용하여 속도를 높인 버전이다.
- 결과적으로 speed-optimal 버전이 LUT 사용량도 더 낮고 Fmax도 높았다.
- 이는 곱셈/누산 연산이 LUT fabric에서 DSP 전용 블록으로 이동했기 때문이다.

### 5.2 FFN pipe

| 항목 | `ffn_pipe.v` post-route 결과 |
|---|---:|
| Slice LUT | 4,477, 8.42% |
| Slice FF | 2,905, 2.73% |
| DSP48E1 | 2, 0.91% |
| BRAM | 0 |
| CARRY4 | 484 |
| WNS @ 100 MHz | +1.842 ns |
| Timing status | MET |
| Estimated Fmax | 약 122.6 MHz |
| Latency | 약 269 cycles, 100 MHz 기준 약 2.69 us |

해석:

- `ffn_pipe.v`는 100 MHz 기준 timing을 만족하였다.
- WNS가 `+1.842 ns`이므로 최악 경로에서도 약 1.842 ns의 timing margin이 남았다.
- LUT 8.42%, FF 2.73%, DSP 0.91%, BRAM 0으로 Zynq-7020 기준 자원 사용량이 과도하지 않다.
- 단, FFN 단독 latency는 2.69 us이므로 전체 1 us 목표와 직접 비교하면 아직 최적화가 필요하다.

## 6. 왜 `ffn.v`가 아니라 `ffn_pipe.v`를 합성했는가

`ffn.v`는 단일사이클 기준 버전이다. 기능 검증에는 유용하지만, 내부에 큰 조합 MAC 연산이 한 cycle에 몰려 있어 Vivado 합성 시 회로 규모가 급격히 커지고 timing closure가 어렵다.

따라서 실제 합성/P&R 대상은 pipelined 구조인 `ffn_pipe.v`로 제한하였다.

`ffn_pipe.v`의 특징은 다음과 같다.

- 32-lane MAC 구조
- FC1은 32차원 입력을 1 chunk로 처리
- FC2는 128차원 입력을 4 chunk로 나누어 누산
- weight ROM을 packed WROM 형태로 구성하여 Vivado area optimization stall을 회피
- pipeline stage를 통해 timing closure 확보

## 7. 보고서에서 강조할 핵심 문장

최종보고서나 발표에서 다음 문장을 사용하면 정확하다.

> 본 프로젝트에서는 전체 QECCT 디코더를 한 번에 FPGA에 통합하기보다, 먼저 핵심 연산 블록인 noise estimator와 Transformer FFN을 Verilog RTL로 구현하였다. 이후 xsim을 통해 INT8 golden model과의 bit-exact 기능 검증을 수행하고, Vivado 2024.2에서 Zynq-7020 타깃 OOC 합성 및 post-route timing/resource 분석을 수행하였다.

결과 문장:

> FFN 블록은 Zynq-7020 기준 post-route 결과에서 LUT 4,477개, FF 2,905개, DSP 2개, BRAM 0개를 사용하였고, 100 MHz constraint에서 WNS +1.842 ns로 timing을 만족하였다.

한계 문장:

> 다만 실물 FPGA 보드 실측은 아직 수행하지 않았으며, 현재 결과는 Vivado 디바이스 모델 기반의 합성/P&R 구현 검증 결과이다. 또한 전체 QECCT decoder top-level 통합은 후속 캡스톤 II 과제로 남아 있다.

## 8. 발표용 30초 설명

> 제가 맡은 FPGA 검증 부분에서는 QECCT Student 모델 전체를 한 번에 FPGA에 올리기보다는, 먼저 핵심 연산 블록인 noise estimator와 Transformer FFN을 Verilog RTL로 구현했습니다. 이후 Python에서 만든 INT8 golden integer model과 RTL 출력을 xsim으로 비교하여 bit-exact 검증을 수행했습니다. 기능 검증 이후에는 Vivado 2024.2에서 Zynq-7020을 타깃으로 out-of-context 합성, 배치, 배선을 수행했고, post-route utilization과 timing report를 확보했습니다. 특히 FFN 블록은 LUT 4,477개, FF 2,905개, DSP 2개, BRAM 0개를 사용했으며, 100 MHz 기준 WNS가 +1.842 ns로 timing을 만족했습니다. 따라서 실물 보드 실측 전 단계에서 핵심 RTL 블록의 FPGA 구현 가능성을 정량적으로 확인했습니다.

## 9. 예상 질문과 답변

### Q1. 실제 FPGA 보드에서 돌린 것인가?

아니다. 실물 보드 실측은 아직 수행하지 않았다. 현재 결과는 Vivado가 제공하는 Zynq-7020 디바이스 모델 기반의 합성, 배치, 배선, timing 분석 결과이다.

### Q2. 그러면 단순 시뮬레이션인가?

단순 시뮬레이션만은 아니다. 기능 검증은 xsim 시뮬레이션으로 수행했고, 그 이후 Vivado에서 실제 FPGA 디바이스 모델에 대해 합성, 배치, 배선까지 수행하였다. 따라서 자원 사용량과 timing closure를 확인한 post-route 구현 검증이다.

### Q3. 전체 QECCT 디코더가 완성된 것인가?

아니다. 캡스톤 I에서는 `noise_estimator`와 `ffn_pipe` 핵심 블록을 우선 검증하였다. embedding, LayerNorm, masked attention, output projection, top-level integration은 후속 과제이다.

### Q4. 결과가 의미 있는가?

의미 있다. Verilog RTL이 golden model과 bit-exact하게 일치했고, Vivado post-route 기준으로 Zynq-7020에서 timing을 만족하는 수치를 확보했기 때문이다. 이는 단순 아이디어가 아니라 FPGA 구현 가능성을 정량적으로 보인 결과이다.
