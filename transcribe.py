# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "mlx-whisper",
#     "pyannote.audio>=4.0",
#     "anthropic",
#     "openai",
#     "python-dotenv",
# ]
# ///
"""Transcribe a video/audio file with speaker identification.

Pipeline:
  1. ffmpeg extracts 16 kHz mono WAV (optionally clipped via --start/--duration)
  2. mlx-whisper transcribes (Apple Silicon optimized)
  3. pyannote/speaker-diarization-community-1 labels who spoke when
  4. Segments are aligned to speakers by maximal time overlap
  5. OpenAI or Anthropic infers real speaker names from conversational context
  6. Writes transcript.json and transcript.md next to the input

Usage:
  uv run transcribe.py gg.mp4 --duration 120
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


KNOWN_NAME_CORRECTIONS = [
    {"from": "Gilmore", "to": "Gillmor"},
    {"from": "Teer", "to": "Teare"},
    {"from": "Tear", "to": "Teare"},
    {"from": "Raddus", "to": "Radice"},
]
KNOWN_NAMES = [
    "Steve Gillmor",
    "Brent Leary",
    "Keith Teare",
    "Frank Radice",
    "Tina Chase",
]
DEFAULT_NAMING_MODELS = {
    "openai": "gpt-5.1",
    "anthropic": "claude-opus-5",
}


def extract_audio(input_path: Path, wav_path: Path, start: float, duration: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(input_path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", "16000", "-vn", str(wav_path)]
    subprocess.run(cmd, check=True)


def transcribe(wav_path: Path, model: str) -> list[dict]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model,
        word_timestamps=True,
        verbose=False,  # False (not None) enables the tqdm progress bar
        # curb repetition hallucinations on disfluent/silent stretches
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
        if seg["text"].strip()
    ]


def diarize(wav_path: Path, hf_token: str, num_speakers: int | None):
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    )
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    with ProgressHook() as hook:
        result = pipeline(str(wav_path), hook=hook, **kwargs)
    # pyannote 4.x community pipelines wrap the Annotation; older ones return it directly
    annotation = getattr(result, "speaker_diarization", result)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def assign_speakers(segments: list[dict], turns: list[dict]) -> None:
    """Label each transcript segment with the speaker whose turns overlap it most."""
    for seg in segments:
        overlaps: dict[str, float] = {}
        for turn in turns:
            ov = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if ov > 0:
                overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0) + ov
        seg["speaker"] = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"


def parse_name_candidates(value: str | None) -> list[str]:
    if not value:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def build_naming_request(
    segments: list[dict], name_candidates: list[str]
) -> tuple[list[str], dict, str, str]:
    labels = sorted({s["speaker"] for s in segments if s["speaker"] != "UNKNOWN"})
    sample_lines = []
    total = 0
    for seg in segments:
        line = f"[{seg['speaker']}] {seg['text']}"
        total += len(line)
        if total > 12000:
            break
        sample_lines.append(line)
    sample = "\n".join(sample_lines)

    schema = {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "name": {"type": "string"},
                        "confident": {"type": "boolean"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["label", "name", "confident", "evidence"],
                    "additionalProperties": False,
                },
            },
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                    },
                    "required": ["from", "to"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["mappings", "corrections"],
        "additionalProperties": False,
    }
    instructions = (
        "You identify speakers in diarized transcripts. Use self-introductions, "
        "being addressed by name, and other conversational context. Only mark a "
        "mapping confident when the transcript itself supports it. When a speaker "
        "is a publicly known person, use the correct real-world spelling of their "
        "name rather than the transcript's phonetic spelling, and if the "
        "transcript misspells a name you map, also emit the corresponding "
        "correction pair, so mappings and corrections agree. Preserve these "
        "known corrections when they appear: Gilmore -> Gillmor, Teer/Tear -> "
        "Teare, and Raddus -> Radice."
    )
    if name_candidates:
        instructions += (
            " The likely speaker roster is: "
            + ", ".join(name_candidates)
            + ". Use these names as candidates for spelling and disambiguation, "
            "but only map a candidate to a diarization label when the transcript "
            "provides supporting evidence."
        )
    prompt = (
        "Map each diarization label to the speaker's real name where the "
        "conversation reveals it. For labels with no evidence, set "
        'confident=false and name="".\n\n'
        "Also list corrections for misspelled proper names in the "
        "transcript text (speaker names, the show's name, and similar - "
        "phonetic mis-transcriptions like 'Teer' for 'Teare'). Each "
        "correction is a {from, to} pair where 'from' is the exact "
        "misspelling as it appears and 'to' is the correct spelling. "
        "Only include corrections you are certain about; never correct "
        "ordinary words.\n\n" + sample
    )
    return labels, schema, instructions, prompt


def infer_names_openai(
    segments: list[dict], model: str, name_candidates: list[str]
) -> tuple[dict[str, str], list[dict]]:
    """Ask OpenAI to map SPEAKER_XX labels and flag misspelled names."""
    import os

    from openai import OpenAI

    labels, schema, instructions, prompt = build_naming_request(segments, name_candidates)
    if not labels:
        return {}, []
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found (checked .env and environment)")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        max_output_tokens=16000,
        instructions=instructions,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "speaker_naming",
                "schema": schema,
                "strict": True,
            }
        },
    )
    if not response.output_text:
        print("OpenAI returned no naming output; keeping generic labels.", file=sys.stderr)
        return {}, []
    return parse_naming_response(json.loads(response.output_text))


def infer_names_anthropic(
    segments: list[dict], model: str, name_candidates: list[str]
) -> tuple[dict[str, str], list[dict]]:
    """Ask Anthropic to map SPEAKER_XX labels and flag misspelled names."""
    import os

    import anthropic

    labels, schema, instructions, prompt = build_naming_request(segments, name_candidates)
    if not labels:
        return {}, []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not found (checked .env and environment)")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=(
            instructions
            + "\nReturn only JSON matching this JSON Schema, with no Markdown fences:\n"
            + json.dumps(schema)
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((block.text for block in response.content if block.type == "text"), "")
    if not text.strip():
        print("Anthropic returned no naming output; keeping generic labels.", file=sys.stderr)
        return {}, []
    return parse_naming_response(json.loads(text))


def infer_names(
    segments: list[dict], provider: str, model: str, name_candidates: list[str]
) -> tuple[dict[str, str], list[dict]]:
    if provider == "anthropic":
        return infer_names_anthropic(segments, model, name_candidates)
    return infer_names_openai(segments, model, name_candidates)


def parse_naming_response(data: dict) -> tuple[dict[str, str], list[dict]]:
    names = {
        m["label"]: m["name"]
        for m in data["mappings"]
        if m["confident"] and m["name"].strip()
    }
    # keep speaker names consistent with the text corrections (e.g. if the
    # corrections fix "Gilmore"→"Gillmor", apply that to the names too)
    import re

    corrections = merge_corrections(KNOWN_NAME_CORRECTIONS, data["corrections"])
    for corr in corrections:
        if corr["from"].strip() and corr["from"] != corr["to"]:
            pattern = re.compile(r"\b" + re.escape(corr["from"]) + r"\b", re.IGNORECASE)
            names = {k: pattern.sub(corr["to"], v) for k, v in names.items()}
    names = {k: normalize_known_name_artifacts(v) for k, v in names.items()}
    return names, corrections


def merge_corrections(*correction_lists: list[dict]) -> list[dict]:
    """Merge correction pairs, keeping first occurrence of each source spelling."""
    merged: list[dict] = []
    seen: set[str] = set()
    for corrections in correction_lists:
        for corr in corrections:
            source = corr["from"].strip()
            target = corr["to"].strip()
            if not source or not target or source.casefold() in seen:
                continue
            seen.add(source.casefold())
            merged.append({"from": source, "to": target})
    return merged


def apply_corrections(segments: list[dict], corrections: list[dict]) -> None:
    """Replace misspelled names in segment text, longest misspelling first."""
    import re

    for corr in sorted(corrections, key=lambda c: -len(c["from"])):
        if not corr["from"].strip() or corr["from"] == corr["to"]:
            continue
        pattern = re.compile(r"\b" + re.escape(corr["from"]) + r"\b", re.IGNORECASE)
        for seg in segments:
            seg["text"] = pattern.sub(corr["to"], seg["text"])
            seg["text"] = normalize_known_name_artifacts(seg["text"])


def normalize_known_name_artifacts(text: str) -> str:
    """Collapse correction artifacts like 'Keith Teare Teare'."""
    import re

    for name in KNOWN_NAMES:
        parts = name.split()
        if len(parts) < 2:
            continue
        last = parts[-1]
        full_then_last = re.compile(
            r"\b" + re.escape(name) + r"\s+" + re.escape(last) + r"\b",
            re.IGNORECASE,
        )
        repeated_full = re.compile(
            r"\b" + re.escape(name) + r"\s+" + re.escape(name) + r"\b",
            re.IGNORECASE,
        )
        text = full_then_last.sub(name, text)
        text = repeated_full.sub(name, text)
    return text


def write_outputs(segments: list[dict], names: dict[str, str], out_dir: Path, source: str) -> None:
    for seg in segments:
        seg["speaker_name"] = names.get(seg["speaker"], seg["speaker"])

    (out_dir / "transcript.json").write_text(
        json.dumps({"source": source, "speakers": names, "segments": segments}, indent=2)
    )

    # display names: first name only, unless two speakers share one
    firsts = [n.split()[0] for n in names.values()]
    display = {
        label: (name.split()[0] if firsts.count(name.split()[0]) == 1 else name)
        for label, name in names.items()
    }

    lines = [f"# Transcript of {source}", ""]
    current, buf = None, []
    def flush():
        if buf:
            lines.append(f"{current}: {' '.join(buf)}")
            lines.append("")
    for seg in segments:
        speaker = display.get(seg["speaker"], seg["speaker"])
        if speaker != current:
            flush()
            current, buf = speaker, []
        buf.append(seg["text"])
    flush()
    (out_dir / "transcript.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe with speaker ID")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--skip-naming", action="store_true", help="skip the naming pass")
    parser.add_argument(
        "--naming-provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="API provider for speaker naming (default: openai)",
    )
    parser.add_argument(
        "--naming-model",
        default=None,
        help="model for the selected naming provider (defaults by provider)",
    )
    parser.add_argument(
        "--name-candidates",
        default=None,
        help="comma-separated likely speaker names for the naming pass",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    parser.add_argument("--duration", type=float, default=None, help="clip length (seconds)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="don't start the transcript viewer after the run")
    args = parser.parse_args()

    load_dotenv()
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not found (checked .env and environment). "
                 "Create one at https://huggingface.co/settings/tokens and accept the "
                 "terms for pyannote/speaker-diarization-community-1.")

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        print("Extracting audio...")
        extract_audio(args.input, wav_path, args.start, args.duration)

        print(f"Transcribing with {args.model}...")
        segments = transcribe(wav_path, args.model)
        print(f"  {len(segments)} segments")

        print("Diarizing with pyannote/speaker-diarization-community-1...")
        turns = diarize(wav_path, hf_token, args.num_speakers)
        print(f"  {len(turns)} speaker turns, "
              f"{len({t['speaker'] for t in turns})} speakers")

    assign_speakers(segments, turns)

    names: dict[str, str] = {}
    if not args.skip_naming:
        naming_model = args.naming_model or DEFAULT_NAMING_MODELS[args.naming_provider]
        name_candidates = parse_name_candidates(args.name_candidates)
        print(f"Inferring speaker names with {args.naming_provider} ({naming_model})...")
        try:
            names, corrections = infer_names(
                segments, args.naming_provider, naming_model, name_candidates
            )
            print(f"  identified: {names}" if names else "  no confident identifications")
            if corrections:
                fixes = ", ".join(f"{c['from']}→{c['to']}" for c in corrections)
                print(f"  spelling fixes: {fixes}")
                apply_corrections(segments, corrections)
        except Exception as e:
            print(f"  naming pass failed ({e}); keeping generic labels", file=sys.stderr)

    write_outputs(segments, names, out_dir, args.input.name)
    print(f"Wrote {out_dir / 'transcript.json'} and {out_dir / 'transcript.md'}")

    if not args.no_viewer:
        launch_viewer()


VIEWER_HOST, VIEWER_PORT = "127.0.0.1", 8787


def launch_viewer() -> None:
    """Start viewer-server.py unless one is already listening; it opens the browser."""
    import socket
    import subprocess

    url = f"http://{VIEWER_HOST}:{VIEWER_PORT}/"
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((VIEWER_HOST, VIEWER_PORT)) == 0:
            print(f"Viewer already running at {url}")
            return

    script = Path(__file__).resolve().parent / "viewer-server.py"
    if not script.exists():
        return
    subprocess.Popen(
        [sys.executable, str(script)],
        cwd=script.parent,
        start_new_session=True,  # survives this process exiting
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started viewer at {url}")


if __name__ == "__main__":
    main()
