# Notebook Inspection Method

Colab'da çalışan notebook'un tüm cell output'larını ve hatalarını okumak için kullanılan Python yöntemi.

## Kullanım

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
        # hata output'u
        if out.get('output_type') == 'error':
            print(f'ERROR: {out.get("ename")}: {out.get("evalue")}')
            for line in out.get('traceback', [])[-6:]:
                print(re.sub(r'\x1b\[[0-9;]*m', '', line))  # ANSI renk kodlarını temizle

        # normal metin output'u (print, stdout)
        else:
            text = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
            if text.strip():
                print(text.strip()[:400])
```

## Nasıl Çalışır

`.ipynb` dosyası aslında bir JSON dosyasıdır. Her cell'in `outputs` alanı şu tipleri içerebilir:

| `output_type` | Ne anlama gelir | İlgili alanlar |
|---|---|---|
| `stream` | `print()` veya `sys.stdout` çıktısı | `text` (list of str) |
| `execute_result` | cell'in return değeri | `data["text/plain"]` |
| `display_data` | `plt.show()`, `Image()` gibi görsel çıktılar | `data["text/plain"]`, `data["image/png"]` |
| `error` | exception | `ename`, `evalue`, `traceback` |

`traceback` içindeki ANSI escape kodları (`\x1b[...m`) terminal renkleri için kullanılır; terminal dışında okunaksız görünür, `re.sub` ile temizlenir.

## Cell Index Mapping

Notebook'taki cell index'i (0-based, tüm cell'ler) ile Colab'daki görsel sıra aynı değildir çünkü markdown cell'leri de sayılır. Code cell numarasını bulmak için:

```python
code_cells = [(i, cell) for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code']
for idx, (i, cell) in enumerate(code_cells):
    src = ''.join(cell['source'])[:60].replace('\n', ' ')
    print(f'code cell #{idx+1}  (nb index {i}): {src}')
```

## Cell Source'u Düzenleme

Bir cell'in içeriğini değiştirmek için:

```python
cell = nb['cells'][TARGET_INDEX]
src = ''.join(cell['source'])

src = src.replace('old_code', 'new_code')

cell['source'] = src.splitlines(keepends=True)
cell['outputs'] = []  # eski output'u temizle (opsiyonel)

with open('notebooks/sample4geo-xai.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

> `splitlines(keepends=True)` her satırı `\n` ile birlikte liste eleman olarak saklar; bu Jupyter'ın beklediği formattır.
