from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("report_assets/figures")
W, H = 1600, 700


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_LABEL = font(26, True)
F_SMALL = font(20)
F_TINY = font(18)


def rounded(draw, xy, fill, outline, radius=22, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color="#1f3b63", width=5):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - sign * 18, y2 - 10), (x2 - sign * 18, y2 + 10)]
    else:
        sign = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 10, y2 - sign * 18), (x2 + 10, y2 - sign * 18)]
    draw.polygon(pts, fill=color)


def text_center(draw, box, lines, fonts=None, fill="#1d2633"):
    x1, y1, x2, y2 = box
    if fonts is None:
        fonts = [F_LABEL] + [F_SMALL] * (len(lines) - 1)
    heights = []
    widths = []
    for line, f in zip(lines, fonts):
        b = draw.textbbox((0, 0), line, font=f)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total_h = sum(heights) + 10 * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, f, tw, th in zip(lines, fonts, widths, heights):
        draw.text((x1 + ((x2 - x1) - tw) / 2, y), line, font=f, fill=fill)
        y += th + 10


def canvas(title, subtitle):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    rounded(d, (22, 22, W - 22, H - 22), "#ffffff", "#d8e0ea", 28, 2)
    d.text((60, 48), title, font=F_TITLE, fill="#1d2633")
    d.text((60, 92), subtitle, font=F_SMALL, fill="#4b5563")
    return img, d


def draw_noise_block():
    img, d = canvas("Noise Estimator Block Diagram", "syndrome 8-bit input -> initial noise feature 9-bit output")
    boxes = [
        ((55, 270, 245, 405), "#eef4ff", "#6f91c5", ["Syndrome", "8-bit"]),
        ((350, 250, 565, 425), "#f5f8fc", "#8fb0d9", ["FC1", "Linear 8 -> 24"]),
        ((670, 250, 885, 425), "#f4fbf7", "#9bc9ad", ["GELU", "LUT approx."]),
        ((990, 250, 1205, 425), "#f5f8fc", "#8fb0d9", ["FC2", "Linear 24 -> 9"]),
        ((1310, 250, 1530, 425), "#f4fbf7", "#9bc9ad", ["Sigmoid", "LUT approx."]),
    ]
    for box, fill, outline, lines in boxes:
        rounded(d, box, fill, outline)
        text_center(d, box, lines)
    for a, b in [((245, 337), (340, 337)), ((565, 337), (660, 337)), ((885, 337), (980, 337)), ((1205, 337), (1300, 337))]:
        arrow(d, a, b)
    d.text((350, 505), "RTL files: noise_estimator.v / noise_estimator_pipe.v", font=F_SMALL, fill="#4b5563")
    d.text((350, 535), "Implementation focus: pipelined INT8 MAC + LUT activation", font=F_SMALL, fill="#4b5563")
    img.save(OUT / "ppt_noise_estimator_block_diagram.png")


def draw_noise_zoom():
    img, d = canvas("Noise Estimator: Pipelined Datapath Zoom", "Main pipeline stages isolated from the dense Vivado schematic")
    boxes = [
        ((60, 245, 235, 385), "#f7f9fc", "#9bb5d8", ["P0", "operand /", "weight select"]),
        ((310, 225, 510, 405), "#eef4ff", "#5f86bf", ["P1", "24 parallel", "INT8 products"]),
        ((585, 225, 785, 405), "#f7f9fc", "#9bb5d8", ["P2-P3", "adder tree", "+ bias"]),
        ((860, 225, 1060, 405), "#eef4ff", "#5f86bf", ["P4", "requant", "multiply"]),
        ((1135, 225, 1335, 405), "#f7f9fc", "#9bb5d8", ["P5", "round / shift", "clamp INT8"]),
    ]
    for box, fill, outline, lines in boxes:
        rounded(d, box, fill, outline)
        text_center(d, box, lines, [F_LABEL] + [F_TINY] * (len(lines) - 1))
    for a, b in [((235, 315), (300, 315)), ((510, 315), (575, 315)), ((785, 315), (850, 315)), ((1060, 315), (1125, 315))]:
        arrow(d, a, b)
    rounded(d, (1020, 485, 1270, 600), "#f4fbf7", "#91c6a4")
    text_center(d, (1020, 485, 1270, 600), ["P6 LUT", "GELU or Sigmoid"], [F_LABEL, F_SMALL])
    arrow(d, (1235, 405), (1235, 480))
    d.text((60, 635), "Key message: pipeline stages and DSP-backed MAC/requantization improve timing.", font=F_SMALL, fill="#4b5563")
    img.save(OUT / "ppt_noise_estimator_datapath_zoom.png")


def draw_ffn_block():
    img, d = canvas("Transformer FFN Block Diagram", "INT8 feed-forward network inside the QECCT Transformer block")
    boxes = [
        ((55, 270, 275, 405), "#eef4ff", "#6f91c5", ["Input", "activation 32 x INT8"]),
        ((390, 250, 640, 425), "#f5f8fc", "#8fb0d9", ["FC1", "Linear 32 -> 128", "32-lane MAC"]),
        ((755, 250, 970, 425), "#f4fbf7", "#9bc9ad", ["GELU", "LUT approx."]),
        ((1085, 250, 1335, 425), "#f5f8fc", "#8fb0d9", ["FC2", "Linear 128 -> 32", "4 chunks / neuron"]),
        ((1430, 270, 1555, 405), "#eef4ff", "#6f91c5", ["Output", "32 x INT8"]),
    ]
    for box, fill, outline, lines in boxes:
        rounded(d, box, fill, outline)
        text_center(d, box, lines, [F_LABEL] + [F_TINY] * (len(lines) - 1))
    for a, b in [((275, 337), (380, 337)), ((640, 337), (745, 337)), ((970, 337), (1075, 337)), ((1335, 337), (1420, 337))]:
        arrow(d, a, b)
    d.text((390, 505), "RTL file: ffn_pipe.v", font=F_SMALL, fill="#4b5563")
    d.text((390, 535), "Implementation focus: 32-lane time-multiplexed MAC + packed WROM", font=F_SMALL, fill="#4b5563")
    img.save(OUT / "ppt_ffn_block_diagram.png")


def draw_ffn_zoom():
    img, d = canvas("Transformer FFN: 32-lane MAC Datapath Zoom", "Structure that keeps ffn_pipe synthesizable")
    rounded(d, (60, 205, 275, 335), "#fff8ec", "#d9b666")
    text_center(d, (60, 205, 275, 335), ["Packed WROM", "32 weights / cycle"], [F_LABEL, F_SMALL])
    rounded(d, (60, 405, 275, 535), "#f7f9fc", "#9bb5d8")
    text_center(d, (60, 405, 275, 535), ["Operands", "xin or hq chunk"], [F_LABEL, F_SMALL])
    rounded(d, (395, 295, 610, 455), "#eef4ff", "#5f86bf")
    text_center(d, (395, 295, 610, 455), ["P1", "32 parallel", "8x8 products"], [F_LABEL, F_TINY, F_TINY])
    rounded(d, (730, 295, 945, 455), "#f7f9fc", "#9bb5d8")
    text_center(d, (730, 295, 945, 455), ["P2-P4", "adder tree", "32 -> 8 -> 2 -> 1"], [F_LABEL, F_TINY, F_TINY])
    rounded(d, (1065, 295, 1280, 455), "#f7f9fc", "#9bb5d8")
    text_center(d, (1065, 295, 1280, 455), ["P5", "chunk", "accumulation"], [F_LABEL, F_TINY, F_TINY])
    rounded(d, (1390, 295, 1560, 455), "#eef4ff", "#5f86bf")
    text_center(d, (1390, 295, 1560, 455), ["P6-P8", "requant", "+ writeback"], [F_LABEL, F_TINY, F_TINY])
    arrow(d, (275, 270), (385, 345))
    arrow(d, (275, 470), (385, 405))
    for a, b in [((610, 375), (720, 375)), ((945, 375), (1055, 375)), ((1280, 375), (1380, 375))]:
        arrow(d, a, b)
    rounded(d, (720, 535, 1190, 615), "#f4fbf7", "#91c6a4")
    text_center(d, (720, 535, 1190, 615), ["FC1: GELU LUT -> hq   |   FC2: yq -> out_flat"], [F_LABEL])
    d.text((60, 645), "Key message: FC2 reduces 128 inputs as four 32-lane chunks, avoiding a huge single-cycle 128-wide MAC.", font=F_SMALL, fill="#4b5563")
    img.save(OUT / "ppt_ffn_mac_zoom.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_noise_block()
    draw_noise_zoom()
    draw_ffn_block()
    draw_ffn_zoom()
    print("rendered ppt block diagrams")
