from __future__ import annotations

import zipfile
from pathlib import Path

import build_result_part_pptx as base


OUT = Path("archive/extracted/capstone_full/결과발표_자료/결과자료ppt/배경_알고리즘_로드맵_추가슬라이드.pptx")


slides = [
    base.slide_xml(
        "연구 배경 및 모델 아키텍처 변경점",
        [
            base.shape_text(3, 0.55, 1.18, 5.8, 5.35, [
                "문제 정의",
                "Surface Code d=3 환경에서는 큐비트 상태 유지를 위해 syndrome decoding이 매우 빠르게 수행되어야 함",
                "실시간 QEC loop를 유지하려면 decoder latency가 1 us 이내여야 함",
                "기존 소프트웨어 MWPM 방식은 수십 us 수준의 지연이 발생하여 온디바이스 실시간 처리에 부적합",
            ], 18, "F7F9FC", "CED6E0", True),
            base.shape_text(4, 6.65, 1.18, 6.1, 5.35, [
                "모델 변경 방향",
                "초기에는 GNN 기반 neural decoder 구조를 검토",
                "하지만 GNN은 그래프 연결 구조와 메모리 접근 패턴이 복잡하여 FPGA timing closure에 불리",
                "최종적으로 Attention 중심의 Transformer 계열 QECCT Student 모델로 전환",
                "정적 연산, 병렬 MAC, INT8 quantization 적용이 쉬워 FPGA/ASIC mapping에 더 적합",
            ], 18, "EEF4FF", "B8C7E6", True),
        ],
        "하드웨어 가속에 적합한 QEC decoder 구조 선택",
    ),
    base.slide_xml(
        "제안 모델의 LER/BER 성능 우수성",
        [
            base.shape_text(3, 0.55, 1.15, 12.2, 0.55, [
                "평가 기준: Surface Code d=3, 물리적 에러율 p 변화에 따른 Logical Error Rate 비교",
            ], 15, "F7F9FC", "CED6E0"),
            base.shape_text(4, 0.75, 2.0, 11.8, 2.75, [
                "물리적 에러율 p        MWPM LER        제안 모델 LER        해석",
                "0.01                  -               0.00000             논리적 오류 미검출 수준",
                "0.05                  -               0.00000             안정적인 오류 정정 성능",
                "0.10                  0.12100         0.00105             높은 오류 환경에서도 큰 폭의 LER 감소",
            ], 18, "FFFFFF", "B8C7E6", True),
            base.shape_text(5, 0.75, 5.15, 11.8, 1.2, [
                "핵심 해석",
                "제안한 CNN/Transformer 계열 모델은 p=0.10의 높은 오류 환경에서도 MWPM 대비 낮은 LER을 보임",
                "따라서 이후 단계에서는 모델 자체의 성능보다, 해당 모델을 INT8 RTL로 변환하고 FPGA에서 구현 가능한지 검증하는 것이 핵심 과제",
            ], 17, "F4FBF7", "BFDCCB", True),
            base.shape_text(6, 0.75, 6.55, 11.8, 0.35, [
                "Note: 본 슬라이드 수치는 팀 제공 데이터 기준이며, 최종 제출 전 원본 실험 파일과 대조 필요",
            ], 10, "FFFFFF", "FFFFFF"),
        ],
    ),
    base.slide_xml(
        "시스템(SoC) 실장 방향성 및 IP 조달",
        [
            base.shape_text(3, 0.55, 1.18, 4.0, 5.35, [
                "현재 캡스톤 I 범위",
                "전체 SoC 완성보다는 AI Engine 핵심 연산 블록 검증에 집중",
                "Noise Estimator 블록과 Transformer FFN 블록의 RTL 구현 및 Vivado OOC 결과 확보",
                "Zynq-7020 기준 자원 사용량, timing closure, latency를 정량 지표로 정리",
            ], 17, "F7F9FC", "CED6E0", True),
            base.shape_text(4, 4.75, 1.18, 3.9, 5.35, [
                "SoC 확장에 필요한 IP",
                "제어용 MCU: VexRiscv 기반 control processor 적용 방향 검토",
                "Memory Controller, AMBA Bus, Clock PLL 등은 삼성 MPW SAFE IP 조달 가능성 협의",
                "AI Engine과 외부 시스템 연결을 위한 AXI4-Stream / AXI-Lite wrapper 필요",
            ], 17, "EEF4FF", "B8C7E6", True),
            base.shape_text(5, 8.85, 1.18, 3.9, 5.35, [
                "Hand-off 관점",
                "가온칩스 2026년 12월 hand-off 일정을 고려",
                "RTL, testbench, timing report, utilization report 중심으로 IP 검토 자료 정리",
                "현재 Zynq-7020 OOC 결과를 기반으로 VU19P 등 상위 FPGA 보드 포팅 검토",
            ], 17, "F4FBF7", "BFDCCB", True),
        ],
    ),
    base.slide_xml(
        "결론 및 캡스톤디자인 II 향후 로드맵",
        [
            base.shape_text(3, 0.55, 1.18, 5.85, 5.35, [
                "캡스톤 I 결론",
                "QECCT Student 모델을 하드웨어 구현 대상으로 선정",
                "핵심 연산 블록을 INT8 RTL로 구현하고 xsim bit-exact 검증 완료",
                "Vivado OOC post-route 결과로 LUT, FF, DSP, BRAM, WNS, Fmax 등 정량 지표 확보",
                "Noise Estimator 블록은 0.46 us로 1 us 목표 달성",
                "FFN 블록은 timing closure는 성공했지만 2.69 us latency로 추가 최적화 필요",
            ], 17, "F7F9FC", "CED6E0", True),
            base.shape_text(4, 6.65, 1.18, 6.1, 5.35, [
                "캡스톤 II 로드맵",
                "1. Embedding RTL 구현",
                "2. LayerNorm RTL 구현",
                "3. Masked Attention RTL 및 softmax 근사 구조 검토",
                "4. Output Projection / Output FC 통합",
                "5. Decoder top-level 통합 및 AXI4-Stream wrapper 추가",
                "6. VU19P 포팅, ASIC gate count / power / lint 등 hand-off 산출물 보강",
            ], 17, "EEF4FF", "B8C7E6", True),
        ],
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    app = base.APP.replace("<Slides>5</Slides>", "<Slides>4</Slides>")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", base.content_types(len(slides)))
        z.writestr("_rels/.rels", base.ROOT_RELS)
        z.writestr("docProps/core.xml", base.CORE.replace("RTL 및 FPGA 검증 결과", "배경 알고리즘 로드맵 추가 슬라이드"))
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", base.presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", base.rels_xml(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", base.SLIDE_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", base.SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", base.SLIDE_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", base.SLIDE_LAYOUT_RELS)
        for i, xml in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", xml)
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>""",
            )
    print(OUT)


if __name__ == "__main__":
    main()
