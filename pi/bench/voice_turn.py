"""Voice A4 benchmark: full STT -> intent -> TTS turn on the Pi.

Pass criterion: median total < 3000ms (VOICE_TURN_BUDGET_MS).
Fail -> set KEYWORD_INTENT_ONLY=1 in production.

Synthesizes test utterances with piper (so we don't need a microphone),
feeds them through Whisper, asks Ollama for intent, then synthesizes the reply.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import wave
from time import perf_counter

import numpy as np

from pi.config import LOCAL_LLM_MODEL, PIPER_VOICE_PATH, STT_MODEL
from pi.models import Detection, DetectionFrame
from pi.services.voice import VoicePipeline

UTTERANCES = [
    "find a bottle",
    "what do you see",
    "status",
    "find the chair",
    "help me",
]

DET = DetectionFrame(
    ts_ms=0,
    boxes=[
        Detection("chair", 0.9, 0.5, 0.5, 0.3, 0.3),
        Detection("person", 0.8, 0.5, 0.5, 0.3, 0.3),
        Detection("bottle", 0.85, 0.5, 0.5, 0.3, 0.3),
    ],
)


def _piper_synth(text: str, out_wav: str) -> None:
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    subprocess.run(
        ["piper", "--model", PIPER_VOICE_PATH, "--output_file", out_wav],
        input=text.encode(),
        check=True,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _read_wav_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw != 2 or ch != 1:
        raise RuntimeError(f"unexpected wav: sw={sw} ch={ch}")
    if sr != 16000:
        ratio = 16000 / sr
        arr = np.frombuffer(frames, dtype=np.int16)
        new_len = int(len(arr) * ratio)
        idx = (np.linspace(0, len(arr) - 1, new_len)).astype(np.int64)
        frames = arr[idx].tobytes()
    return frames


def main() -> int:
    print(f"models: stt={STT_MODEL}  llm={LOCAL_LLM_MODEL}  voice={PIPER_VOICE_PATH}", flush=True)

    voice = VoicePipeline()
    print("warming whisper + ollama ...", flush=True)
    t = perf_counter()
    voice.warmup()
    print(f"  warmup {perf_counter() - t:.2f}s", flush=True)

    print("synthesizing test utterances with piper ...", flush=True)
    cached: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for u in UTTERANCES:
            wav_path = os.path.join(tmp, f"{abs(hash(u))}.wav")
            _piper_synth(u, wav_path)
            cached[u] = _read_wav_pcm16(wav_path)
            print(f"  {u!r}: {len(cached[u]) / 32000:.2f}s of audio", flush=True)

    print("\nrunning end-to-end turns ...", flush=True)
    rows = []
    for u in UTTERANCES:
        pcm = cached[u]
        t0 = perf_counter()
        text = voice.transcribe(pcm)
        t_stt = perf_counter()
        result = voice.derive_intent(text, DET)
        t_int = perf_counter()
        reply_pcm = b""
        if result.reply:
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                _piper_synth(result.reply, wav.name)
                reply_pcm = _read_wav_pcm16(wav.name)
        t_tts = perf_counter()

        stt_ms = (t_stt - t0) * 1000
        int_ms = (t_int - t_stt) * 1000
        tts_ms = (t_tts - t_int) * 1000
        total_ms = (t_tts - t0) * 1000
        rows.append((u, text, result.intent, result.target, result.reply,
                     stt_ms, int_ms, tts_ms, total_ms,
                     len(reply_pcm) / 32000))
        print(
            f"  total={total_ms:7.0f}ms  stt={stt_ms:6.0f} intent={int_ms:6.0f} tts={tts_ms:6.0f}  "
            f"intent={result.intent:13s} target={str(result.target):10s}  "
            f"heard={text!r}  reply={result.reply[:40]!r}",
            flush=True,
        )

    totals = [r[8] for r in rows]
    print()
    print(f"frames    : {len(totals)}")
    print(f"median ms : {statistics.median(totals):.0f}")
    print(f"max ms    : {max(totals):.0f}")
    print(f"min ms    : {min(totals):.0f}")
    print()
    if statistics.median(totals) < 3000:
        print("VERDICT: PASS  (median < 3000ms = A4 voice budget). Local-first OK.")
        return 0
    print("VERDICT: FAIL  (median >= 3000ms). Set KEYWORD_INTENT_ONLY=1.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
