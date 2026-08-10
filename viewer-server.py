#!/usr/bin/env python3
"""Serve a local XMLUI transcript viewer."""

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
ALL_SPEAKERS = "__all__"


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_transcript(path: Path) -> dict:
    data = json.loads(path.read_text())
    segments = data.get("segments", [])
    turns = []
    for segment in segments:
        speaker = segment.get("speaker_name") or segment.get("speaker") or "UNKNOWN"
        text = segment.get("text", "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = segment.get("end", turns[-1]["end"])
            turns[-1]["text"] = f"{turns[-1]['text']} {text}"
            turns[-1]["segmentCount"] += 1
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "time": format_time(segment.get("start", 0)),
                    "text": text,
                    "segmentCount": 1,
                }
            )
    speakers = sorted({turn["speaker"] for turn in turns})
    duration = max((segment.get("end", 0) for segment in segments), default=0)
    return {
        "source": data.get("source", path.name),
        "duration": duration,
        "durationLabel": format_time(duration),
        "segmentCount": len(segments),
        "turnCount": len(turns),
        "speakers": speakers,
        "speakerOptions": [{"value": ALL_SPEAKERS, "label": "All speakers"}]
        + [{"value": speaker, "label": speaker} for speaker in speakers],
        "turns": turns,
    }


def render_index() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Transcript Viewer</title>
  <script src="https://cdn.xmlui.org/xmlui.js"></script>
</head>
<body>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    transcript_path: Path

    def do_GET(self) -> None:
        self.route_request(send_body=True)

    def do_HEAD(self) -> None:
        self.route_request(send_body=False)

    def route_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_bytes(render_index(), "text/html; charset=utf-8", send_body)
            return
        if path in {"/Main.xmlui", "/viewer.xmlui"}:
            self.send_file(ROOT / "viewer.xmlui", "application/xml; charset=utf-8", send_body)
            return
        if path == "/api/transcript":
            payload = json.dumps(load_transcript(self.transcript_path)).encode()
            self.send_bytes(payload, "application/json; charset=utf-8", send_body)
            return
        if path == "/api/raw-transcript":
            self.send_file(self.transcript_path, "application/json; charset=utf-8", send_body)
            return
        candidate = (ROOT / path.lstrip("/")).resolve()
        if ROOT in candidate.parents and candidate.is_file():
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_file(candidate, content_type, send_body)
            return
        self.send_error(404, "Not found")

    def send_file(self, path: Path, content_type: str, send_body: bool) -> None:
        if not path.exists():
            self.send_error(404, f"{path.name} not found")
            return
        self.send_bytes(path.read_bytes(), content_type, send_body)

    def send_bytes(self, body: bytes, content_type: str, send_body: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def open_browser_later(url: str) -> None:
    time.sleep(0.5)
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the transcript viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--transcript", type=Path, default=ROOT / "transcript.json")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    transcript_path = args.transcript.resolve()
    if not transcript_path.exists():
        raise SystemExit(f"{transcript_path} not found")

    ViewerHandler.transcript_path = transcript_path
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving transcript viewer at {url}")
    print(f"Transcript: {transcript_path}")
    if not args.no_open:
        threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping transcript viewer")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
