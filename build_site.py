"""Assemble the website from the page template and the pipeline outputs, so every
number on the page comes straight from outputs/*.json.

Writes:
  site/index.html  page body (as used for the Claude artifact)
  index.html       standalone page at the repo root, served by GitHub Pages
Usage: python build_site.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S, O = os.path.join(HERE, "site"), os.path.join(HERE, "outputs")


def js(name):
    return json.dumps(json.load(open(os.path.join(O, name))), separators=(",", ":"))


head = open(os.path.join(S, "_head.html"), encoding="utf-8").read()
body = open(os.path.join(S, "_body.html"), encoding="utf-8").read()
for key, f in [("__RESULTS__", "results.json"), ("__SHAP__", "shap_results.json"),
               ("__EXAMPLE__", "example_explanation.json"), ("__WEEK__", "tg_week.json"),
               ("__DQ__", "data_quality.json")]:
    body = body.replace(key, js(f))
page = head.split("<header")[0] + body
open(os.path.join(S, "index.html"), "w", encoding="utf-8").write(page)

shell = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
         '<meta name="description" content="Federated, graph-aware, explainable 3-hour-ahead load forecasting '
         'on real Grid-India hourly demand for six Southern Region states.">\n'
         '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
         'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}'
         'img{max-width:100%}[hidden]{display:none!important}</style>\n')
i = page.index("</style>") + len("</style>")
standalone = shell + page[:i] + "\n</head>\n<body>\n" + page[i:] + "\n</body>\n</html>\n"
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(standalone)
print("wrote site/index.html and index.html")
