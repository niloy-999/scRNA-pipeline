"""
Start the app with a 1024 MB upload limit.

    python run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "1024"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

from streamlit.web import cli as stcli


def main() -> int:
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.maxUploadSize=1024",
        "--server.maxMessageSize=1024",
        "--server.fileWatcherType=poll",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())