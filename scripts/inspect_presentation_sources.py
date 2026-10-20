"""只读提取本次演讲稿、建设方案的 OOXML 内容，辅助 V4 审阅。"""

from pathlib import Path
import json
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/presentation_v4"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main", "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ROOT / "IBERO_工业级双臂具身智能操作基准平台演讲PPT_v3.pptx") as archive:
        result = []
        pres = ET.fromstring(archive.read("ppt/presentation.xml"))
        print("SLIDE SIZE", pres.find("p:sldSz", NS).attrib)
        names = sorted((n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.search(r"(\d+)\.xml", n).group(1)))
        for i, name in enumerate(names, 1):
            root = ET.fromstring(archive.read(name))
            shapes = []
            for shape in root.findall(".//p:sp", NS):
                paragraphs = ["".join(t.text or "" for t in p.findall(".//a:t", NS)) for p in shape.findall(".//a:p", NS)]
                shapes.append({"text": "\n".join(paragraphs), "fonts": [el.attrib for el in shape.findall(".//a:rPr", NS)], "geometry": [el.attrib for el in shape.findall(".//a:xfrm/*", NS)]})
            notes_name = f"ppt/notesSlides/notesSlide{i}.xml"
            notes = ""
            if notes_name in archive.namelist():
                notes = "\n".join(t.text or "" for t in ET.fromstring(archive.read(notes_name)).findall(".//a:t", NS))
            result.append({"slide": i, "shapes": shapes, "notes": notes})
        (OUT / "v3_structure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        text = "\n\n".join(f"=== SLIDE {s['slide']} ===\n" + "\n".join(sh['text'] for sh in s['shapes']) + "\nNOTES:\n" + s['notes'] for s in result)
        (OUT / "v3_text.txt").write_text(text, encoding="utf-8")
    with ZipFile(ROOT / "IBERO__工业级双臂具身智能操作基准平台建设方案.docx") as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        text = "\n".join("".join(t.text or "" for t in p.findall(".//w:t", NS)) for p in root.findall(".//w:p", NS))
        (OUT / "proposal_text.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
