import sys, os
# .streamlit/start_fastapi.py -> project root
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)
import uvicorn
uvicorn.run("web_app:app", host="0.0.0.0", port=8000, log_level="info")