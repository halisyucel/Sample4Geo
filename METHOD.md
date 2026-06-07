# Notebook Inspection Method

Python-based method for reading all cell outputs and errors from a running Colab notebook.

## Usage

```python
import json, re

with open('notebooks/sample4geo-xai.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    if not outputs:
        continue

    print(f'\n=== CELL {i} ({cell["cell_type"]}) ===')

    for out in outputs:
        # error output
        if out.get('output_type') == 'error':
            print(f'ERROR: {out.get("ename")}: {out.get("evalue")}')
            for line in out.get('traceback', [])[-6:]:
                print(re.sub(r'\x1b\[[0-9;]*m', '', line))  # strip ANSI color codes

        # text output (print, stdout)
        else:
            text = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
            if text.strip():
                print(text.strip()[:400])
```

## How It Works

A `.ipynb` file is a JSON document. Each cell has an `outputs` field that can contain the following types:

| `output_type` | Meaning | Relevant fields |
|---|---|---|
| `stream` | `print()` or `sys.stdout` output | `text` (list of str) |
| `execute_result` | cell return value | `data["text/plain"]` |
| `display_data` | visual output (`plt.show()`, `Image()`, etc.) | `data["text/plain"]`, `data["image/png"]` |
| `error` | exception | `ename`, `evalue`, `traceback` |

The `traceback` field contains ANSI escape codes (`\x1b[...m`) for terminal colors. These are unreadable outside a terminal and are stripped with `re.sub`.

## Cell Index Mapping

The cell index in the notebook JSON (0-based, all cells) differs from the visual order in Colab because markdown cells are also counted. To map code cell numbers:

```python
code_cells = [(i, cell) for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code']
for idx, (i, cell) in enumerate(code_cells):
    src = ''.join(cell['source'])[:60].replace('\n', ' ')
    print(f'code cell #{idx+1}  (nb index {i}): {src}')
```

## Editing Cell Source

To modify a cell's content directly in the JSON:

```python
cell = nb['cells'][TARGET_INDEX]
src = ''.join(cell['source'])

src = src.replace('old_code', 'new_code')

cell['source'] = src.splitlines(keepends=True)
cell['outputs'] = []  # optionally clear stale outputs

with open('notebooks/sample4geo-xai.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

> `splitlines(keepends=True)` stores each line as a list element with its trailing `\n`, which is the format Jupyter expects.
