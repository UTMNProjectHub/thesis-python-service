from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Iterable, List, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.api.core.config import settings


PathLike = Union[str, Path]
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


_SUBSCRIPT_CHARS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₐ": "a",
    "ₑ": "e",
    "ₕ": "h",
    "ᵢ": "i",
    "ⱼ": "j",
    "ₖ": "k",
    "ₗ": "l",
    "ₘ": "m",
    "ₙ": "n",
    "ₒ": "o",
    "ₚ": "p",
    "ᵣ": "r",
    "ₛ": "s",
    "ₜ": "t",
    "ᵤ": "u",
    "ᵥ": "v",
    "ₓ": "x",
}
_SUPERSCRIPT_CHARS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁱ": "i",
    "ⁿ": "n",
}
_SUBSCRIPT_RE = re.compile(f"[{''.join(re.escape(ch) for ch in _SUBSCRIPT_CHARS)}]+")
_SUPERSCRIPT_RE = re.compile(f"[{''.join(re.escape(ch) for ch in _SUPERSCRIPT_CHARS)}]+")

# TODO: Сделать подгрузку шрифтов
_FONT_FAMILIES = [
    {
        "regular": str(PROJECT_ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf"),
        "bold": str(PROJECT_ROOT / "assets" / "fonts" / "NotoSans-Bold.ttf"),
        "italic": str(PROJECT_ROOT / "assets" / "fonts" / "NotoSans-Italic.ttf"),
        "bold_italic": str(PROJECT_ROOT / "assets" / "fonts" / "NotoSans-BoldItalic.ttf"),
    },
    {
        "regular": "C:/Windows/Fonts/arial.ttf",
        "bold": "C:/Windows/Fonts/arialbd.ttf",
        "italic": "C:/Windows/Fonts/ariali.ttf",
        "bold_italic": "C:/Windows/Fonts/arialbi.ttf",
    },
    {
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "bold_italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    },
]


def _register_unicode_font() -> str:
    env_family = {
        "regular": settings.pdf_font_regular,
        "bold": settings.pdf_font_bold,
        "italic": settings.pdf_font_italic,
        "bold_italic": settings.pdf_font_bold_italic,
    }
    families = [env_family] if all(env_family.values()) else []
    families.extend(_FONT_FAMILIES)

    for family in families:
        regular = Path(family["regular"])
        bold = Path(family["bold"])
        italic = Path(family["italic"])
        bold_italic = Path(family["bold_italic"])
        if not all(path.exists() for path in (regular, bold, italic, bold_italic)):
            continue

        pdfmetrics.registerFont(TTFont("LectureFont", str(regular)))
        pdfmetrics.registerFont(TTFont("LectureFont-Bold", str(bold)))
        pdfmetrics.registerFont(TTFont("LectureFont-Italic", str(italic)))
        pdfmetrics.registerFont(TTFont("LectureFont-BoldItalic", str(bold_italic)))
        pdfmetrics.registerFontFamily(
            "LectureFont",
            normal="LectureFont",
            bold="LectureFont-Bold",
            italic="LectureFont-Italic",
            boldItalic="LectureFont-BoldItalic",
        )
        logger.info("Registered PDF unicode font family regular=%s", regular)
        return "LectureFont"

    logger.warning("Unicode TTF font was not found; Cyrillic PDF output can be unreadable")
    return "Helvetica"


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "LectureBody",
        parent=base["BodyText"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        spaceAfter=7,
        firstLineIndent=0,
        alignment=TA_LEFT,
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "LectureH1",
            parent=body,
            fontName=f"{font_name}-Bold" if font_name != "Helvetica" else "Helvetica-Bold",
            fontSize=20,
            leading=26,
            spaceBefore=4,
            spaceAfter=14,
            textColor=colors.HexColor("#1f2933"),
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "LectureH2",
            parent=body,
            fontName=f"{font_name}-Bold" if font_name != "Helvetica" else "Helvetica-Bold",
            fontSize=16,
            leading=22,
            spaceBefore=12,
            spaceAfter=10,
            textColor=colors.HexColor("#25364a"),
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "LectureH3",
            parent=body,
            fontName=f"{font_name}-Bold" if font_name != "Helvetica" else "Helvetica-Bold",
            fontSize=13,
            leading=18,
            spaceBefore=10,
            spaceAfter=7,
            textColor=colors.HexColor("#34495e"),
            keepWithNext=True,
        ),
        "quote": ParagraphStyle(
            "LectureQuote",
            parent=body,
            leftIndent=12,
            rightIndent=8,
            borderColor=colors.HexColor("#d0d7de"),
            borderWidth=1,
            borderPadding=8,
            textColor=colors.HexColor("#4b5563"),
            backColor=colors.HexColor("#f7f9fb"),
        ),
        "code": ParagraphStyle(
            "LectureCode",
            parent=body,
            fontName=font_name,
            fontSize=9,
            leading=12,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor("#d0d7de"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f6f8fa"),
            wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "LectureList",
            parent=body,
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=4,
        ),
    }


def _inline_markdown(text: str, font_name: str) -> str:
    text = html.escape(text.strip())
    text = _replace_unicode_scripts(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"`([^`]+)`", rf'<font name="{font_name}" backColor="#f2f4f7">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<i>\1</i>", text)
    return text


def _replace_unicode_scripts(text: str) -> str:
    def replace_sub(match: re.Match[str]) -> str:
        value = "".join(_SUBSCRIPT_CHARS[ch] for ch in match.group(0))
        return f"<sub>{value}</sub>"

    def replace_super(match: re.Match[str]) -> str:
        value = "".join(_SUPERSCRIPT_CHARS[ch] for ch in match.group(0))
        return f"<super>{value}</super>"

    text = _SUBSCRIPT_RE.sub(replace_sub, text)
    return _SUPERSCRIPT_RE.sub(replace_super, text)


def _setext_heading_level(line: str) -> int | None:
    if re.match(r"^=+\s*$", line):
        return 1
    if re.match(r"^-+\s*$", line):
        return 2
    return None


def _horizontal_rule(line: str) -> bool:
    return bool(
        re.match(r"^([-*_])(?:\s*\1){2,}\s*$", line)
        or re.match(r"^={3,}\s*$", line)
    )


def _trailing_decorated_heading(line: str) -> tuple[str, int] | None:
    match = re.match(r"^(.+?)\s+([=-])\2{3,}\s*$", line)
    if not match:
        return None

    title = match.group(1).strip()
    if not title:
        return None
    return title, 2 if match.group(2) == "=" else 3


def _is_block_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^#{1,6}\s+", stripped)
        or re.match(r"^[-*+]\s+", stripped)
        or re.match(r"^\d+[.)]\s+", stripped)
        or stripped.startswith(">")
        or stripped.startswith("```")
        or _horizontal_rule(stripped)
    )


def _list_flowable(items: Iterable[str], ordered: bool, styles: dict[str, ParagraphStyle], font_name: str) -> ListFlowable:
    flowables = [
        ListItem(Paragraph(_inline_markdown(item, font_name), styles["list"]), leftIndent=14)
        for item in items
    ]
    params = {
        "bulletType": "1" if ordered else "bullet",
        "leftIndent": 18,
        "bulletFontName": font_name,
        "bulletFontSize": 10,
    }
    if ordered:
        params["start"] = "1"
    return ListFlowable(flowables, **params)


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.match(r"^:?-{3,}:?$", cell or "") for cell in cells)


def _parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle], font_name: str) -> Table:
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]
    data = [
        [Paragraph(_inline_markdown(cell, font_name), styles["body"]) for cell in row]
        for row in normalized
    ]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_to_flowables(markdown: str, styles: dict[str, ParagraphStyle], font_name: str) -> List:
    lines = markdown.replace("\r\n", "\n").split("\n")
    flowables: List = []
    paragraph: List[str] = []
    i = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph if part.strip())
        paragraph.clear()
        if text:
            flowables.append(Paragraph(_inline_markdown(text, font_name), styles["body"]))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            code_lines: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            flowables.append(Preformatted("\n".join(code_lines), styles["code"]))
            flowables.append(Spacer(1, 5))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)), 3)
            flowables.append(Paragraph(_inline_markdown(heading.group(2), font_name), styles[f"h{level}"]))
            i += 1
            continue

        setext_level = _setext_heading_level(stripped)
        if setext_level and paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            paragraph.clear()
            flowables.append(Paragraph(_inline_markdown(text, font_name), styles[f"h{min(setext_level, 3)}"]))
            i += 1
            continue

        decorated_heading = _trailing_decorated_heading(stripped)
        if decorated_heading:
            flush_paragraph()
            title, level = decorated_heading
            flowables.append(Paragraph(_inline_markdown(title, font_name), styles[f"h{level}"]))
            i += 1
            continue

        if _horizontal_rule(stripped):
            flush_paragraph()
            flowables.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=colors.HexColor("#d0d7de"),
                    spaceBefore=6,
                    spaceAfter=10,
                )
            )
            i += 1
            continue

        if "|" in stripped and i + 1 < len(lines) and _is_table_separator(lines[i + 1].strip()):
            flush_paragraph()
            table_rows = [_parse_table_row(stripped)]
            i += 2
            while i < len(lines):
                current = lines[i].strip()
                if not current or "|" not in current:
                    break
                table_rows.append(_parse_table_row(current))
                i += 1
            flowables.append(_table_flowable(table_rows, styles, font_name))
            flowables.append(Spacer(1, 7))
            continue

        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if bullet or ordered:
            flush_paragraph()
            ordered_mode = bool(ordered)
            items: List[str] = []
            while i < len(lines):
                current = lines[i].strip()
                match = re.match(r"^\d+[.)]\s+(.+)$", current) if ordered_mode else re.match(r"^[-*+]\s+(.+)$", current)
                if not match:
                    break
                items.append(match.group(1))
                i += 1
            flowables.append(_list_flowable(items, ordered_mode, styles, font_name))
            flowables.append(Spacer(1, 5))
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote = stripped.lstrip(">").strip()
            flowables.append(Paragraph(_inline_markdown(quote, font_name), styles["quote"]))
            i += 1
            continue

        if _is_block_start(stripped):
            flush_paragraph()
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return flowables


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def markdown_to_pdf(md_path: PathLike, pdf_path: PathLike) -> str:
    """
    Convert UTF-8 Markdown to a styled PDF with a Unicode font.
    Supports headings, paragraphs, bold/italic, inline code, bullet/numbered lists,
    setext headings, separators, blockquotes, fenced code blocks, and page breaks.
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = md_path.read_text(encoding="utf-8")
    font_name = _register_unicode_font()
    styles = _build_styles(font_name)
    story = _markdown_to_flowables(markdown, styles, font_name)

    if not story:
        story = [Paragraph("", styles["body"])]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=md_path.stem,
        author="Quizy",
    )
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return str(pdf_path)
