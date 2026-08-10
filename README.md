# gg

Transcription with speaker identification for `gg.mp4`. The pipeline runs
locally on Apple Silicon: `mlx-whisper` transcribes, `pyannote` figures out
who spoke when, and an OpenAI or Anthropic naming pass replaces
`SPEAKER_00`-style labels with real names inferred from the conversation.

## How the pipeline works

`transcribe.py` chains five stages. The first four run entirely on your
machine; only the last one (optional) calls an external API.

### 1. Audio extraction — ffmpeg

The video's audio track (48 kHz stereo Opus) is decoded to a temporary
16 kHz mono WAV file, the input format both models expect. `--start` and
`--duration` clip here, so a partial run never decodes more than it needs.

### 2. Transcription — mlx-whisper

[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
is Whisper (OpenAI's speech-recognition model) running on
[MLX](https://github.com/ml-explore/mlx), Apple's array framework for
Apple Silicon, so inference runs on the Mac's GPU. The default model,
`whisper-large-v3-turbo`, is a distilled large-v3 — near large-v3 accuracy
at several times the speed. Output is a list of segments, each with text
and start/end timestamps. Two settings curb Whisper's classic
repetition-hallucination failure mode (`condition_on_previous_text=False`,
`hallucination_silence_threshold`).

Whisper transcribes *what* was said, but has no concept of *who* said it —
that's the next stage.

### 3. Diarization — pyannote

[pyannote.audio](https://github.com/pyannote/pyannote-audio) answers "who
spoke when." The
[speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline detects speech regions, computes a voice embedding (a numeric
fingerprint of each voice), and clusters the embeddings so each distinct
voice becomes a speaker. Output is a list of turns like *SPEAKER_02 spoke
from 12.4 s to 19.1 s*. It doesn't know names — labels are arbitrary. The
model is gated on Hugging Face, which is why setup requires a token and
accepting its terms. Runs on the GPU via MPS when available.

### 4. Alignment

Pure Python, no models: each Whisper segment is assigned the speaker whose
diarization turns overlap it the most (segment and turn boundaries never
match exactly, so maximal-overlap is the standard heuristic). Consecutive
segments from the same speaker are later merged into readable turns in the
Markdown output.

### 5. Speaker naming and spelling repair — OpenAI or Anthropic

The one non-local, optional stage (`--skip-naming` omits it). A sample of
the labeled transcript goes to the selected naming provider, which does two things:

- **Names the speakers** from conversational evidence — self-introductions,
  people addressing each other ("okay Tina let's go"), host introductions
  ("From the Duchy of Palo Alto is Keith Teare"). Only confidently
  identified labels are renamed; the rest stay `SPEAKER_NN`.
- **Repairs name spellings** in the transcript text. Whisper spells names
  phonetically ("Gilmore", "Teer", "Raddus"); the naming provider returns
  `{from, to}` correction pairs for proper names it is certain about, and
  the script applies them as case-insensitive whole-word replacements
  across the full transcript. The script also seeds the known Gillmor Gang
  corrections from the original Claude pass (`Gillmor`, `Teare`, `Radice`)
  so those names are fixed consistently before any extra corrections inferred
  fresh from each recording.

## Setup

### 1. Install system tools

```sh
brew install ffmpeg uv
```

(`uv` manages the script's Python environment automatically — no venv or
`pip install` needed.)

### 2. Get a Hugging Face token (for the diarization model)

1. Create an account at [huggingface.co](https://huggingface.co) if needed.
2. Visit [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and accept the model's user conditions (a short form).
3. Create an access token at [Settings → Access Tokens](https://huggingface.co/settings/tokens).
   A fine-grained token works; make sure it includes
   **"Read access to contents of all public gated repos you can access."**

### 3. Create `.env`

In the repo root:

```sh
echo HF_TOKEN=hf_your_token_here > .env
```

No quotes around the value — a stray quote becomes part of the token and
Hugging Face will reject it with a 401. (`.env` is gitignored.)

### 4. Get and set a naming API key

The speaker-naming pass defaults to Anthropic with `claude-opus-5`. Get a
Claude API key at [platform.claude.com](https://platform.claude.com), then
either export it:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

or add a second line to `.env`:

```sh
ANTHROPIC_API_KEY=sk-ant-...
```

To skip naming entirely, run with `--skip-naming` — you'll get
`SPEAKER_00`-style labels instead of names.

OpenAI is also supported if you have an OpenAI key
([platform.openai.com](https://platform.openai.com/api-keys)); export
`OPENAI_API_KEY` or add it to `.env`, then:

```sh
uv run transcribe.py gg.mp4 --duration 180 --naming-provider openai
```

For better speaker naming, pass a known roster and override the naming model
when needed:

```sh
uv run transcribe.py gg.mp4 --duration 600 --num-speakers 5 \
  --name-candidates "Steve Gillmor, Brent Leary, Keith Teare, Frank Radice, Tina Chase"

uv run transcribe.py gg.mp4 --duration 600 --naming-model claude-opus-5
```

## Run

The examples assume a local `gg.mp4` in the repo root. In this workspace,
`gg.mp4` was downloaded with `yt-dlp`; the current copy is about 375 MB
(358 MiB on disk) and is intentionally untracked. You can use a larger
original/local recording instead if it has a better audio track.

```sh
# Quick test: first two minutes
uv run transcribe.py gg.mp4 --duration 120

# First five minutes
uv run transcribe.py gg.mp4 --duration 300

# Full file
uv run transcribe.py gg.mp4
```

The first run downloads the Python dependencies and models (a few GB); after
that, runs start immediately. Outputs land next to the input:

- `transcript.json` — segments with start/end times, speaker labels, and names
- `transcript.md` — readable transcript with `**Name** [hh:mm:ss]:` turns

## View

The viewer starts automatically at the end of a transcription run: if
nothing is listening on port 8787, `transcribe.py` spawns
`viewer-server.py`, which opens your browser on the fresh transcript. If
the server is already running, the new transcript is live there as soon
as it's written. Pass `--no-viewer` to skip this for headless runs.

To start the viewer manually (e.g. to revisit an old transcript):

```sh
python3 viewer-server.py
```

The server opens your browser automatically at `http://127.0.0.1:8787/`.
It serves a small XMLUI app from `viewer.xmlui`, reads `transcript.json`,
and provides source/duration summary, speaker chips, speaker filtering,
search, and a turn-by-turn transcript view.

The viewer can also launch transcription runs: the **Run transcription**
card takes an optional duration and speaker count, starts
`uv run transcribe.py gg.mp4` on the server, streams live status and log
output into the page (server-sent events feeding an XMLUI `PushSource`),
and reloads the transcript when the run completes. One run at a time;
a second request while one is running is rejected. Use `--no-open` if you want to
start the server without opening a browser:

```sh
python3 viewer-server.py --no-open
```

## Logs

Every run — terminal or viewer-launched — writes `logs/run-<timestamp>.log`
(gitignored): the exact command line, resolved configuration, per-stage
timings and counts, and the naming outcome (identified speakers and spelling
fixes, or a prominent `NAMING FAILED:` line with the reason). When a
transcript looks wrong, start there.

## Options

| Flag | Meaning |
|---|---|
| `--model` | mlx-whisper model repo (default `mlx-community/whisper-large-v3-turbo`) |
| `--num-speakers N` | Tell the diarizer exactly how many speakers to find |
| `--skip-naming` | Skip the naming pass (no OpenAI or Anthropic API key needed) |
| `--naming-provider openai\|anthropic` | Choose the naming API provider (default `anthropic`) |
| `--naming-model MODEL` | Override the naming model (defaults: `gpt-5.1` for OpenAI, `claude-opus-5` for Anthropic) |
| `--name-candidates NAMES` | Comma-separated likely speaker names to guide naming without forcing mappings |
| `--output-dir DIR` | Write outputs somewhere other than next to the input |
| `--no-viewer` | Don't start the transcript viewer after the run |
| `--start S` | Start the clip at S seconds |
| `--duration S` | Only process S seconds of audio |
