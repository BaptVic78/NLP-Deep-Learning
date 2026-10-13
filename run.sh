#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec venv/bin/python -m streamlit run app.py --server.port "${PORT:-8504}" --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
