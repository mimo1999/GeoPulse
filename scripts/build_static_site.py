"""Assemble site/ for GitHub Pages: the Streamlit app running in the browser via stlite.

    python scripts/export_static_assets.py   # refresh assets/*.csv (needs Postgres)
    python scripts/build_static_site.py
    python -m http.server -d site 8600       # preview
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
STLITE = "0.85.1"

# Fetched on first use in the browser (see streamlit_app/data_source.py), so not mounted at startup.
REMOTE_ASSETS = {"gi_daily.csv", "gi_events.csv", "gi_counterparts.csv"}

CODE_FILES = [
    "streamlit_app/app.py",
    "streamlit_app/data_source.py",
    "streamlit_app/ui.py",
    "streamlit_app/pages/01_country_drilldown.py",
    "streamlit_app/pages/02_global_intelligence.py",
    "data/iso3_to_fips.py",
]
ALL_ASSETS = sorted(p.name for p in (ROOT / "assets").iterdir() if p.is_file())
MOUNTED = CODE_FILES + [f"assets/{n}" for n in ALL_ASSETS if n not in REMOTE_ASSETS]

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GeoPulse Activity Monitor</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.css" />
</head>
<body>
  <div id="root"></div>
  <script type="module">
    const files = __FILES__;
    // Lets the Python side fetch large assets lazily, relative to wherever the site is hosted.
    files["static_config.py"] = `BASE_URL = "${new URL("./", location.href).href}"
`;
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.js";
    mount({
      entrypoint: "streamlit_app/app.py",
      requirements: ["plotly"],
      files,
    }, document.getElementById("root"));
  </script>
</body>
</html>
"""


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    for rel in CODE_FILES + [f"assets/{n}" for n in ALL_ASSETS]:
        dest = SITE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    files = {rel: {"url": f"./{rel}"} for rel in MOUNTED}
    (SITE / "index.html").write_text(
        HTML.replace("__V__", STLITE).replace("__FILES__", json.dumps(files, indent=6)), encoding="utf-8"
    )
    (SITE / ".nojekyll").touch()
    print(f"built {SITE} ({len(MOUNTED)} mounted files, {len(ALL_ASSETS)} assets)")


if __name__ == "__main__":
    main()
