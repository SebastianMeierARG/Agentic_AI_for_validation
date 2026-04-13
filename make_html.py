"""
Convert any Markdown file to a styled HTML file and open it in the browser.

Usage:
    python make_html.py                   # converts USER_MANUAL.
    md → USER_MANUAL.html
    python make_html.py CLAUDE.md         # converts any .md file
"""

import sys
import os
import webbrowser
import mistune

CSS = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #24292e;
    background: #f6f8fa;
    margin: 0;
    padding: 40px 20px;
  }
  .container {
    max-width: 900px;
    margin: 0 auto;
    background: #fff;
    border: 1px solid #d1d5da;
    border-radius: 6px;
    padding: 48px 56px;
  }
  h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
  h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 2em; }
  h3 { font-size: 1.2em; margin-top: 1.6em; }
  h4 { font-size: 1em; margin-top: 1.4em; }
  code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 85%;
    background: #f0f3f6;
    border-radius: 3px;
    padding: 0.2em 0.4em;
  }
  pre {
    background: #f0f3f6;
    border-radius: 6px;
    padding: 16px;
    overflow-x: auto;
    font-size: 85%;
  }
  pre code { background: none; padding: 0; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 14px;
  }
  th {
    background: #f0f3f6;
    font-weight: 600;
    text-align: left;
    padding: 8px 12px;
    border: 1px solid #d1d5da;
  }
  td {
    padding: 8px 12px;
    border: 1px solid #d1d5da;
    vertical-align: top;
  }
  tr:nth-child(even) td { background: #f9fafb; }
  blockquote {
    margin: 0;
    padding: 0 1em;
    color: #6a737d;
    border-left: 4px solid #dfe2e5;
  }
  a { color: #0366d6; }
  hr { border: none; border-top: 1px solid #eaecef; margin: 2em 0; }
  li { margin: 0.3em 0; }
  .toc { background: #f6f8fa; border: 1px solid #eaecef; border-radius: 6px; padding: 16px 24px; margin-bottom: 2em; }
  .toc h2 { margin-top: 0; font-size: 1em; color: #6a737d; text-transform: uppercase; letter-spacing: 0.05em; border: none; }
  .badge {
    display: inline-block;
    background: #0366d6;
    color: white;
    font-size: 11px;
    font-weight: 600;
    border-radius: 12px;
    padding: 2px 8px;
    margin-left: 6px;
    vertical-align: middle;
  }
</style>
"""


def convert(md_path: str) -> str:
    html_path = os.path.splitext(md_path)[0] + ".html"

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    body = mistune.html(md_text)
    title = os.path.splitext(os.path.basename(md_path))[0].replace("_", " ")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {CSS}
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


if __name__ == "__main__":
    md_file = sys.argv[1] if len(sys.argv) > 1 else "USER_MANUAL.md"

    if not os.path.exists(md_file):
        print(f"Error: '{md_file}' not found.")
        sys.exit(1)

    out = convert(md_file)
    print(f"Converted: {md_file} -> {out}")
    webbrowser.open(f"file:///{os.path.abspath(out)}")
