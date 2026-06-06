from __future__ import annotations

import zipfile
from pathlib import Path

import build_context_roadmap_pptx as context
import build_result_part_pptx as base
import build_result_part_pptx as result


OUT = Path("archive/extracted/capstone_full/결과발표_자료/결과자료ppt/ROQET_RTL_FPGA_검증_종합발표.pptx")


# Order: background/model performance -> RTL/FPGA verification -> SoC/roadmap.
slides = [
    context.slides[0],
    context.slides[1],
    result.slides[0],
    result.slides[1].replace("RTL 설계 - noise_estimator", "RTL 설계 - Noise Estimator 블록"),
    result.slides[2].replace("RTL 설계 - ffn_pipe", "RTL 설계 - Transformer FFN 블록"),
    result.slides[3].replace("FPGA 합성 결과 - noise_estimator", "FPGA 합성 결과 - Noise Estimator 블록"),
    result.slides[4].replace("FPGA 합성 결과 - ffn_pipe", "FPGA 합성 결과 - Transformer FFN 블록"),
    context.slides[2],
    context.slides[3],
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    app = base.APP.replace("<Slides>5</Slides>", f"<Slides>{len(slides)}</Slides>")
    core = base.CORE.replace("RTL 및 FPGA 검증 결과", "ROQET RTL FPGA 검증 종합발표")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", base.content_types(len(slides)))
        z.writestr("_rels/.rels", base.ROOT_RELS)
        z.writestr("docProps/core.xml", core)
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
