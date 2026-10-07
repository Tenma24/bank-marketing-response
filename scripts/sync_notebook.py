"""Update notebook cells from the percent-format Python source.

Outputs survive only when the corresponding code cell is unchanged.
After changing analysis code, execute run_notebook.py to refresh all evidence.
"""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def parsed_cells():
    text = (ROOT / "src" / "bank_marketing.py").read_text(encoding="utf-8")
    cells = []
    for section in re.split(r"^# %%", text, flags=re.MULTILINE):
        if not section.strip():
            continue
        if section.startswith(" [markdown]"):
            lines = section.split("\n", 1)[1].splitlines()
            content = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                                for line in lines).strip()
            cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            cells.append(nbformat.v4.new_code_cell(section.strip()))
    return cells

if __name__ == "__main__":
    path = ROOT / "notebooks" / "bank_marketing.ipynb"
    notebook = nbformat.read(path, as_version=4)
    new_cells = parsed_cells()
    for index, cell in enumerate(new_cells):
        if index < len(notebook.cells):
            old = notebook.cells[index]
            if old.cell_type == cell.cell_type and old.source == cell.source:
                new_cells[index] = old
    notebook.cells = new_cells
    nbformat.write(notebook, path)
    print(f"Synchronized {len(new_cells)} notebook cells.")
