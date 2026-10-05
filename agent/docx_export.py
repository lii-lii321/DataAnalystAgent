import re
from pathlib import Path

from docx import Document
from docx.shared import Inches

INLINE_MARKUP_RE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")
MONO_FONT = "Consolas"


def _inline_runs(text: str) -> list[tuple[str, bool, bool]]:
    """Split a line into (text, bold, monospace) runs, stripping ** and ` markers."""
    runs: list[tuple[str, bool, bool]] = []
    pos = 0
    for match in INLINE_MARKUP_RE.finditer(text):
        if match.start() > pos:
            runs.append((text[pos : match.start()], False, False))
        if match.group(1) is not None:
            runs.append((match.group(1), True, False))
        else:
            runs.append((match.group(2), False, True))
        pos = match.end()
    if pos < len(text):
        runs.append((text[pos:], False, False))
    return runs


def _fill_paragraph(paragraph, text: str) -> None:
    for part, bold, mono in _inline_runs(text):
        run = paragraph.add_run(part)
        run.bold = bold
        if mono:
            run.font.name = MONO_FONT


def markdown_to_docx(md: str, out_path, image_dir=None) -> str:
    """Render the analysis report (headings, bullets, tables, images) into a .docx file."""
    doc = Document()
    lines = md.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(set(cell) <= set("-: ") for cell in cells):
                    rows.append(cells)
                index += 1
            if rows:
                width = max(len(row) for row in rows)
                table = doc.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                for row_no, row in enumerate(rows):
                    for col_no in range(width):
                        cell = table.cell(row_no, col_no)
                        _fill_paragraph(cell.paragraphs[0], row[col_no] if col_no < len(row) else "")
            continue

        if stripped.startswith("![") and "](" in stripped:
            match = re.search(r"\]\(([^)]+)\)", stripped)
            if match:
                candidate = Path(image_dir or ".") / Path(match.group(1)).name
                if candidate.exists():
                    doc.add_picture(str(candidate), width=Inches(5.8))
                index += 1
                continue

        if stripped.startswith("### "):
            _fill_paragraph(doc.add_heading("", level=2), stripped[4:])
        elif stripped.startswith("## "):
            _fill_paragraph(doc.add_heading("", level=1), stripped[3:])
        elif stripped.startswith("# "):
            _fill_paragraph(doc.add_heading("", level=0), stripped[2:])
        elif stripped.startswith("- "):
            _fill_paragraph(doc.add_paragraph("", style="List Bullet"), stripped[2:])
        elif stripped:
            _fill_paragraph(doc.add_paragraph(""), stripped)
        index += 1

    doc.save(out_path)
    return str(out_path)
