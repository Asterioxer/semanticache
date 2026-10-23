"""CLI entry point for semanticache demo."""

import argparse
import os
import sys
import webbrowser


def main():
    parser = argparse.ArgumentParser(prog="semanticache")
    subparsers = parser.add_subparsers(dest="command")
    demo_parser = subparsers.add_parser("demo", help="Launch the local demo dashboard")
    demo_parser.add_argument("--host", default="0.0.0.0")
    demo_parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    if args.command == "demo":
        url = f"http://localhost:{args.port}"
        print(f"Starting semanticache demo at {url}")
        # Open browser after a short delay so server has time to start
        import threading
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

        import uvicorn
        uvicorn.run(
            "semanticache.demo.server:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
