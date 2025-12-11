from __future__ import annotations

from pathlib import Path
from typing import Union
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


PathLike = Union[str, Path]


def markdown_to_pdf(md_path: PathLike, pdf_path: PathLike) -> str:
    """
    Простейшая конвертация markdown-файла в PDF.

    md_path  — путь к исходному .md
    pdf_path — путь к целевому .pdf

    Возвращает str-путь к созданному PDF.
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)

    text = md_path.read_text(encoding="utf-8")

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    left_margin = 40
    top_margin = 40
    bottom_margin = 40
    line_height = 14

    y = height - top_margin

    max_chars = 90

    for line in text.splitlines():
        if not line.strip():
            # пустая строка = отступ
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                y = height - top_margin
            continue

        wrapped_lines = textwrap.wrap(line, width=max_chars)
        for wl in wrapped_lines:
            c.drawString(left_margin, y, wl)
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                y = height - top_margin

    c.save()
    return str(pdf_path)
