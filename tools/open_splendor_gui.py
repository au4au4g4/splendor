#!/usr/bin/env python3
"""Open the browser-based Splendor AI interface.

The graphical game is maintained by cestpasphoto and runs entirely in the
browser.  This helper exists so users of this repository have an obvious
``python tools/open_splendor_gui.py`` entry point for the GUI workflow.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

OFFICIAL_GUI_URL = "https://cestpasphoto.github.io/splendor.html"
REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PAGE = REPO_ROOT / "gui" / "splendor_gui.html"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve local files without noisy per-request logging."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the Splendor graphical AI UI.")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Open the official hosted GUI directly instead of the local helper page.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the local helper page over http://localhost so the iframe can load reliably.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port used with --serve.",
    )
    return parser.parse_args()


def open_official() -> None:
    print(f"Opening official Splendor GUI: {OFFICIAL_GUI_URL}")
    webbrowser.open(OFFICIAL_GUI_URL)


def open_local_file() -> None:
    print(f"Opening local GUI helper: {WRAPPER_PAGE}")
    webbrowser.open(WRAPPER_PAGE.resolve().as_uri())


def serve_wrapper(port: int) -> None:
    if not WRAPPER_PAGE.exists():
        raise FileNotFoundError(f"Missing GUI helper page: {WRAPPER_PAGE}")

    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(REPO_ROOT), **kwargs)
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/gui/splendor_gui.html"
        print(f"Serving GUI helper at {url}")
        print("Press Ctrl+C to stop the local helper server.")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped GUI helper server.")


def main() -> None:
    args = parse_args()
    if args.official:
        open_official()
    elif args.serve:
        serve_wrapper(args.port)
    else:
        open_local_file()


if __name__ == "__main__":
    main()
