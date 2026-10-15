"""Build index.html (GitHub Pages) and preview.html (Artifact) from src/."""
import json, pathlib, re
root = pathlib.Path(__file__).parent
src = root / 'src'
tpl = (src / 'template.html').read_text(encoding='utf-8')
data = {
    'startups': (root / 'data' / '50_Startups.csv').read_text(encoding='utf-8'),
    'tsmc': (root / 'data' / '2330_TW.csv').read_text(encoding='utf-8'),
}
body = (tpl.replace('/*CSS*/', (src / 'style.css').read_text(encoding='utf-8'))
           .replace('/*DATA*/', 'window.DEFAULT_DATA = ' + json.dumps(data, ensure_ascii=False) + ';')
           .replace('/*ML*/', (src / 'ml.js').read_text(encoding='utf-8'))
           .replace('/*APP*/', (src / 'app.js').read_text(encoding='utf-8')))
head_end = body.index('<header')
full = ('<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + body[:head_end] + '</head>\n<body>\n' + body[head_end:] + '</body>\n</html>\n')
(root / 'index.html').write_text(full, encoding='utf-8')
(root.parent / 'preview.html').write_text(body, encoding='utf-8')
print('index.html', len(full), 'bytes')
