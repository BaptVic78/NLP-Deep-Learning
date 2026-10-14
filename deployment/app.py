"""Cloud entry point for the existing Streamlit app."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
runpy.run_path(str(root / 'app.py'), run_name='__main__')
