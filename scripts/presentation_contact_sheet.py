"""将 PowerPoint 导出的审阅图排成缩略图目录（仅审阅产物）。"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    slides = sorted(args.directory.glob("slide_*.png"))
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    for start in range(0, len(slides), 12):
        batch = slides[start:start + 12]
        sheet = Image.new("RGB", (1600, 990), "#101923")
        draw = ImageDraw.Draw(sheet)
        for index, source in enumerate(batch):
            x, y = index % 3 * 533, index // 3 * 247
            picture = Image.open(source)
            picture.thumbnail((520, 220))
            sheet.paste(picture, (x, y + 25))
            draw.text((x + 8, y), source.stem, font=font, fill="white")
        sheet.save(args.directory / f"contact_sheet_{start // 12 + 1}.png")


if __name__ == "__main__":
    main()
