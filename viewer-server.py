#!/usr/bin/env python3
"""Serve a local XMLUI transcript viewer."""

import argparse
import json
import mimetypes
import shutil
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
ALL_SPEAKERS = "__all__"

# Bright categorical palette: (background, label) per speaker, cycled by index.
SPEAKER_PALETTE = [
    ("#E5484D", "white"),    # red
    ("#0090FF", "white"),    # blue
    ("#30A46C", "white"),    # green
    ("#F76B15", "white"),    # orange
    ("#8E4EC6", "white"),    # violet
    ("#FFC53D", "#3D2E00"),  # amber (dark label for contrast)
    ("#E93D82", "white"),    # pink
    ("#12A594", "white"),    # teal
]


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
    speaker_colors = {
        speaker: {
            "background": SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)][0],
            "label": SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)][1],
        }
        for i, speaker in enumerate(speakers)
    }
    duration = max((segment.get("end", 0) for segment in segments), default=0)
    return {
        "source": data.get("source", path.name),
        "duration": duration,
        "durationLabel": format_time(duration),
        "segmentCount": len(segments),
        "turnCount": len(turns),
        "speakers": speakers,
        "speakerColors": speaker_colors,
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
  <script>
    // Host bridge between this server and the XMLUI app's <PushSource>.
    //
    // The server side: GET /api/run/events is a server-sent events (SSE)
    // endpoint. It holds the HTTP response open and writes a JSON snapshot
    // of the transcription job ({state, command, logTail, ...}) every time
    // the status changes -- no client polling.
    //
    // The browser side: EventSource is the built-in SSE client. Each
    // server write fires onmessage with the JSON payload as text.
    //
    // The XMLUI side: viewer.xmlui declares
    //   <PushSource id="runStatus" subscribe="{window.subscribeRunStatus}">
    // On mount, XMLUI calls subscribe(emit) exactly once, handing us the
    // emit callback. Every emit(value) below becomes a reactive update:
    // any binding that reads runStatus.value (the status badge, the log
    // tail, the Run button's enabled state) re-renders automatically, and
    // a ChangeListener watching runStatus.value.state refetches the
    // transcript when a run completes.
    //
    // The returned function is the cleanup contract: XMLUI calls it if
    // the PushSource unmounts, closing the SSE connection.
    window.subscribeRunStatus = (emit) => {
      const es = new EventSource("/api/run/events");
      es.onmessage = (e) => emit(JSON.parse(e.data));
      return () => es.close();
    };
  </script>
  <script src="https://cdn.xmlui.org/xmlui.js"></script>
</head>
<body>
</body>
</html>
"""


# ---- transcription runs -----------------------------------------------------

SERVER_LOG_LOCK = threading.Lock()


def server_log(msg: str) -> None:
    """Journal run-request lifecycle events to logs/viewer-server.log."""
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    with SERVER_LOG_LOCK:
        with (logs_dir / "viewer-server.log").open("a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg)


PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def have_key(name: str) -> bool:
    """True if the key is set in the server environment or in .env."""
    import os

    if os.environ.get(name):
        return True
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(name + "=") and line.split("=", 1)[1].strip():
                return True
    return False


JOB_LOCK = threading.Lock()
JOB = {"state": "idle", "command": "", "log": [], "returncode": None, "started": 0.0, "finished": 0.0, "logFile": ""}


def job_snapshot() -> dict:
    with JOB_LOCK:
        if JOB["state"] == "running":
            elapsed = time.time() - JOB["started"]
        elif JOB["started"]:
            elapsed = JOB["finished"] - JOB["started"]
        else:
            elapsed = 0
        return {
            "state": JOB["state"],
            "command": JOB["command"],
            "returncode": JOB["returncode"],
            "elapsedLabel": format_time(elapsed),
            "logTail": "\n".join(JOB["log"][-25:]),
            "logFile": JOB["logFile"],
        }


def start_run(options: dict) -> tuple[bool, str]:
    server_log(f"run requested: {json.dumps(options)}")
    with JOB_LOCK:
        if JOB["state"] == "running":
            server_log("run rejected: a run is already in progress")
            return False, "a transcription run is already in progress"
        uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
        cmd = [uv, "run", "transcribe.py", "gg.mp4", "--no-viewer"]
        if options.get("duration"):
            cmd += ["--duration", str(float(options["duration"]))]
        if options.get("start"):
            cmd += ["--start", str(float(options["start"]))]
        if options.get("numSpeakers"):
            cmd += ["--num-speakers", str(int(options["numSpeakers"]))]
        provider = options.get("namingProvider")
        if provider == "skip":
            cmd += ["--skip-naming"]
        elif provider in PROVIDER_KEYS:
            key = PROVIDER_KEYS[provider]
            if not have_key(key):
                server_log(f"run rejected: {key} missing for provider '{provider}'")
                return False, (f"{key} not found in the server environment or .env; "
                               "add it or choose another naming option")
            cmd += ["--naming-provider", provider]
        elif provider:
            server_log(f"run option ignored: unknown namingProvider {provider!r}")
        try:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
        except OSError as e:
            server_log(f"run failed to spawn: {e}")
            return False, str(e)
        JOB.update(state="running", command=" ".join(cmd[2:]), log=[],
                   returncode=None, started=time.time(), finished=0.0, logFile="")
    server_log(f"run started: {' '.join(cmd)}")
    threading.Thread(target=watch_run, args=(proc,), daemon=True).start()
    return True, "started"


def watch_run(proc: subprocess.Popen) -> None:
    for raw in proc.stdout:
        # tqdm redraws with \r; keep only the final state of each line
        line = raw.rstrip("\n").split("\r")[-1].strip()
        if not line:
            continue
        with JOB_LOCK:
            if line.startswith("Run log: "):
                JOB["logFile"] = line.removeprefix("Run log: ")
            JOB["log"].append(line)
            del JOB["log"][:-400]
    code = proc.wait()
    with JOB_LOCK:
        JOB["state"] = "done" if code == 0 else "failed"
        JOB["returncode"] = code
        JOB["finished"] = time.time()
        state, log_file = JOB["state"], JOB["logFile"]
    server_log(f"run finished: state={state} returncode={code} logFile={log_file}")


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
        if path == "/api/run/status":
            payload = json.dumps(job_snapshot()).encode()
            self.send_bytes(payload, "application/json; charset=utf-8", send_body)
            return
        if path == "/api/run/events":
            self.stream_run_events()
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

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                options = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                options = {}
            ok, message = start_run(options if isinstance(options, dict) else {})
            body = json.dumps({"ok": ok, "message": message}).encode()
            self.send_response(200 if ok else 409)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not found")

    def stream_run_events(self) -> None:
        """Server-sent events: push a status snapshot whenever it changes."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                payload = json.dumps(job_snapshot())
                if payload != last:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    last = payload
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            return

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
