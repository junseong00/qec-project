from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT = Path("archive/extracted/capstone_full/결과발표_자료/결과자료ppt/박준성_RTL_FPGA_결과발표_파트.pptx")

EMU_PER_IN = 914400
SLIDE_W = 13.333333
SLIDE_H = 7.5


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def tx_body(lines: list[str], font_size: int = 20, bold_first: bool = False) -> str:
    paras = []
    for i, line in enumerate(lines):
        bold = ' b="1"' if bold_first and i == 0 else ""
        paras.append(
            f"""
            <a:p>
              <a:r>
                <a:rPr lang="ko-KR" sz="{font_size * 100}"{bold}/>
                <a:t>{esc(line)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
      <p:txBody>
        <a:bodyPr wrap="square"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>"""


def shape_text(shape_id: int, x: float, y: float, w: float, h: float, lines: list[str],
               font_size: int = 20, fill: str = "FFFFFF", line: str = "D9DEE7",
               bold_first: bool = False) -> str:
    xemu = int(x * EMU_PER_IN)
    yemu = int(y * EMU_PER_IN)
    wemu = int(w * EMU_PER_IN)
    hemu = int(h * EMU_PER_IN)
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{xemu}" y="{yemu}"/><a:ext cx="{wemu}" cy="{hemu}"/></a:xfrm>
        <a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
      {tx_body(lines, font_size, bold_first)}
    </p:sp>"""


def title_shape(title: str, subtitle: str = "") -> str:
    lines = [title] + ([subtitle] if subtitle else [])
    return shape_text(2, 0.55, 0.35, 12.25, 0.8, lines, font_size=26, fill="F7F9FC", line="F7F9FC", bold_first=True)


def footer() -> str:
    return shape_text(90, 0.55, 7.02, 12.25, 0.22, ["ROQET Capstone Design I | RTL & FPGA Verification"], font_size=8, fill="FFFFFF", line="FFFFFF")


def slide_xml(title: str, boxes: list[str], subtitle: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title_shape(title, subtitle)}
      {''.join(boxes)}
      {footer()}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


slides = [
    slide_xml(
        "RTL 및 FPGA 검증 결과",
        [
            shape_text(3, 0.8, 1.55, 5.75, 3.9, [
                "발표 범위",
                "1. RTL 설계 - noise_estimator",
                "2. RTL 설계 - ffn_pipe",
                "3. FPGA 합성 결과 - noise_estimator",
                "4. FPGA 합성 결과 - ffn_pipe",
            ], 21, "EEF4FF", "B8C7E6", True),
            shape_text(4, 6.85, 1.55, 5.7, 3.9, [
                "핵심 메시지",
                "전체 SoC 완성 결과가 아니라, AI Engine 핵심 연산 블록의 RTL 기능 검증과 Vivado OOC 구현 가능성을 정량 확인함",
                "xsim bit-exact 검증 + post-route utilization/timing report 기반으로 발표",
            ], 19, "F7F9FC", "CED6E0", True),
        ],
        "noise_estimator / ffn_pipe 중심",
    ),
    slide_xml(
        "RTL 설계 - noise_estimator",
        [
            shape_text(3, 0.55, 1.25, 4.0, 5.3, [
                "모듈 명세",
                "파일: rtl/noise_estimator.v",
                "파일: rtl/noise_estimator_pipe.v",
                "입력: syndrome 8-bit",
                "출력: noise feature INT8",
                "구조: FC -> GELU/LUT -> FC -> Sigmoid/LUT",
            ], 17, "F7F9FC", "CED6E0", True),
            shape_text(4, 4.8, 1.25, 3.7, 5.3, [
                "설계 포인트",
                "Area-minimal 버전과 speed-optimal pipeline 버전을 모두 구현",
                "Pipeline 버전은 DSP48E1을 활용해 LUT 사용량과 timing을 개선",
                "BRAM 없이 distributed LUT-ROM 기반으로 파라미터 저장",
            ], 17, "EEF4FF", "B8C7E6", True),
            shape_text(5, 8.75, 1.25, 4.0, 5.3, [
                "기능 검증",
                "검증 방식: PyTorch INT8 golden model vs Verilog RTL xsim",
                "20 vectors",
                "180/180 output bytes 일치",
                "mismatch 0",
                "결론: bit-exact PASS",
            ], 17, "F4FBF7", "BFDCCB", True),
        ],
    ),
    slide_xml(
        "RTL 설계 - ffn_pipe",
        [
            shape_text(3, 0.55, 1.25, 4.0, 5.3, [
                "모듈 명세",
                "파일: rtl/ffn_pipe.v",
                "역할: Transformer block 내부 FFN",
                "구조: Linear(32->128) -> GELU -> Linear(128->32)",
                "정밀도: INT8 고정소수점",
            ], 17, "F7F9FC", "CED6E0", True),
            shape_text(4, 4.8, 1.25, 3.7, 5.3, [
                "설계 포인트",
                "32-lane time-multiplexed MAC",
                "FC1은 32차원 입력을 1 chunk로 처리",
                "FC2는 128차원 입력을 4 chunks로 누산",
                "Packed WROM 구조로 Vivado area optimization stall 완화",
            ], 17, "EEF4FF", "B8C7E6", True),
            shape_text(5, 8.75, 1.25, 4.0, 5.3, [
                "기능 검증",
                "검증 방식: INT8 golden model vs RTL xsim",
                "100 vectors",
                "3200/3200 output bytes 일치",
                "mismatch 0",
                "결론: bit-exact PASS",
            ], 17, "F4FBF7", "BFDCCB", True),
        ],
    ),
    slide_xml(
        "FPGA 합성 결과 - noise_estimator",
        [
            shape_text(3, 0.55, 1.18, 12.2, 0.62, [
                "Target: Zynq-7020 xc7z020clg400-1 | Tool: Vivado 2024.2 | Flow: OOC synth + place & route | Clock: 100 MHz",
            ], 15, "F7F9FC", "CED6E0"),
            shape_text(4, 0.55, 2.05, 5.8, 4.5, [
                "Area-minimal",
                "Slice LUT: 3,310 (6.22%)",
                "Slice FF: 293 (0.28%)",
                "DSP48E1: 0",
                "BRAM: 0",
                "Fmax: 17.9 MHz",
                "Latency: 약 2.01 us",
            ], 18, "FFF8EC", "E4C47A", True),
            shape_text(5, 6.7, 2.05, 5.8, 4.5, [
                "Speed-optimal pipeline",
                "Slice LUT: 912 (1.71%)",
                "Slice FF: 340 (0.32%)",
                "DSP48E1: 26 (11.82%)",
                "BRAM: 0",
                "Fmax: 105.3 MHz",
                "Latency: 약 0.46 us",
                "결론: 1 us 목표 달성",
            ], 18, "F4FBF7", "BFDCCB", True),
        ],
    ),
    slide_xml(
        "FPGA 합성 결과 - ffn_pipe",
        [
            shape_text(3, 0.55, 1.18, 12.2, 0.62, [
                "Target: Zynq-7020 xc7z020clg400-1 | Tool: Vivado 2024.2 | Flow: OOC post-route | Clock: 100 MHz",
            ], 15, "F7F9FC", "CED6E0"),
            shape_text(4, 0.75, 2.0, 5.45, 4.7, [
                "Post-route utilization",
                "Slice LUT: 4,477 (8.42%)",
                "Slice FF: 2,905 (2.73%)",
                "DSP48E1: 2 (0.91%)",
                "BRAM: 0",
                "CARRY4: 484",
            ], 19, "EEF4FF", "B8C7E6", True),
            shape_text(5, 6.65, 2.0, 5.95, 4.7, [
                "Timing 및 해석",
                "WNS @ 100 MHz: +1.842 ns",
                "Timing status: MET",
                "Estimated Fmax: 약 122.6 MHz",
                "Latency: 약 269 cycles, 2.69 us",
                "결론: timing은 만족, latency는 1 us 목표 초과",
                "개선 방향: MAC lane 확장 및 scheduling 최적화",
            ], 18, "F7F9FC", "CED6E0", True),
        ],
    ),
]


def rels_xml(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
    ]
    for i in range(slide_count):
        rels.append(
            f'<Relationship Id="rId{i + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def presentation_xml(slide_count: int) -> str:
    sld_ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>' for i in range(slide_count)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def content_types(slide_count: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(slide_count):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>"""


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

SLIDE_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles/>
</p:sldMaster>"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>RTL 및 FPGA 검증 결과</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""

APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>5</Slides>
</Properties>"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/core.xml", CORE)
        z.writestr("docProps/app.xml", APP)
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)
        for i, xml in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", xml)
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>""",
            )
    print(OUT)


if __name__ == "__main__":
    main()
