"""
Start the FastAPI Quantization Server
=======================================

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --host 0.0.0.0 --port 8000

Then visit http://localhost:8000/docs for interactive API documentation.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting LLM Quantization API at http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
