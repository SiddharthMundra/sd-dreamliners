"""Voice pipeline: PCM16 audio -> STT -> intent -> reply -> TTS.

Two intent paths, switchable at runtime:
  - Ollama LLM intent (default; A4 gate-pass)
  - KEYWORD_INTENT_ONLY (A4 gate-fail or KEYWORD_INTENT_ONLY=1)

Heavy deps (faster_whisper, ollama, piper) are imported lazily so the module
loads on a laptop without them installed.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import threading
from time import perf_counter
from typing import Callable

import numpy as np

from pi.config import (
    LOCAL_LLM_HOST,
    LOCAL_LLM_MODEL,
    PIPER_VOICE_PATH,
    STT_COMPUTE_TYPE,
    STT_MODEL,
    STT_SAMPLE_RATE,
    USE_KEYWORD_INTENT_ONLY,
)
from pi.models import DetectionFrame, IntentResult, VoiceTurn

log = logging.getLogger(__name__)


VALID_INTENTS = {"IDLE", "SEEKING", "WHATDOYOUSEE", "STATUS", "EMERGENCY"}

_EMERGENCY_RE = re.compile(
    r"\b(?:help|sos|emergency|i('?m| am) hurt|i('?ve| have) fallen|i fell)\b",
    re.I,
)

_LLM_INTENT_ALIASES = {
    "find": "SEEKING",
    "search": "SEEKING",
    "look": "SEEKING",
    "locate": "SEEKING",
    "see": "WHATDOYOUSEE",
    "what": "WHATDOYOUSEE",
    "describe": "WHATDOYOUSEE",
    "ok": "STATUS",
    "help": "EMERGENCY",
    "sos": "EMERGENCY",
    "alert": "EMERGENCY",
}


_KEYWORDS: list[tuple[re.Pattern, Callable[[re.Match], IntentResult]]] = [
    (re.compile(r"\b(?:help|sos|emergency|i('?m| am) hurt|i fell)\b", re.I),
     lambda m: IntentResult(intent="EMERGENCY", reply="Sending help now.")),
    (re.compile(r"\bfind (?:the |a |an )?(.+)", re.I),
     lambda m: IntentResult(intent="SEEKING", target=m.group(1).strip(),
                            reply=f"Looking for a {m.group(1).strip()}.")),
    (re.compile(r"\bwhat (?:do you|can you) see\b", re.I),
     lambda m: IntentResult(intent="WHATDOYOUSEE")),
    (re.compile(r"\b(?:status|how are you)\b", re.I),
     lambda m: IntentResult(intent="STATUS", reply="All systems running.")),
]


class VoicePipeline:
    def __init__(self) -> None:
        self._stt = None
        self._stt_lock = threading.Lock()
        self.last_turn_ms: dict[str, float] = {}

    def warmup(self) -> None:
        self._ensure_stt()
        if not USE_KEYWORD_INTENT_ONLY:
            try:
                self._ollama_chat("ping", "")
            except Exception as e:
                log.warning("ollama warmup failed: %s", e)

    def transcribe(self, pcm16_bytes: bytes) -> str:
        if not pcm16_bytes:
            return ""
        self._ensure_stt()
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with self._stt_lock:
            segments, _ = self._stt.transcribe(audio, language="en", beam_size=1)  # type: ignore[union-attr]
            text = " ".join(s.text for s in segments).strip()
        return text

    def derive_intent(self, transcript: str, det: DetectionFrame) -> IntentResult:
        if not transcript:
            return IntentResult(intent="IDLE")
        sos = _EMERGENCY_RE.search(transcript)
        if sos:
            return IntentResult(intent="EMERGENCY", reply="Sending help now.")
        if USE_KEYWORD_INTENT_ONLY:
            return self._keyword_intent(transcript, det)
        try:
            return self._llm_intent(transcript, det)
        except Exception as e:
            log.warning("llm intent failed (%s), falling back to keywords", e)
            return self._keyword_intent(transcript, det)

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._piper_speak(text)
        except Exception as e:
            log.warning("tts failed: %s", e)

    def _ensure_stt(self) -> None:
        if self._stt is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        log.info("loading whisper model: %s (%s)", STT_MODEL, STT_COMPUTE_TYPE)
        self._stt = WhisperModel(STT_MODEL, device="cpu", compute_type=STT_COMPUTE_TYPE)

    def _llm_intent(self, transcript: str, det: DetectionFrame) -> IntentResult:
        det_summary = ", ".join(sorted({b.cls for b in det.boxes})) or "nothing"
        system = (
            'Output JSON only: {"intent":"<X>","target":"<obj>|null","reply":"<one short sentence>"}. '
            "intent MUST be one of: IDLE, SEEKING, WHATDOYOUSEE, STATUS, EMERGENCY. "
            "Use SEEKING when the user wants to find something. "
            "Use WHATDOYOUSEE when the user asks what is around. "
            "Use STATUS for greetings or status checks. "
            "Use EMERGENCY for help or fall calls. "
            "Use IDLE otherwise. "
            f"Camera sees: {det_summary}."
        )
        raw = self._ollama_chat(transcript, system)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._keyword_intent(transcript, det)
        intent = self._normalize_intent(parsed.get("intent", ""))
        if intent is None:
            return self._keyword_intent(transcript, det)
        return IntentResult(
            intent=intent,
            target=parsed.get("target"),
            reply=parsed.get("reply", "") or "",
        )

    @staticmethod
    def _normalize_intent(raw: str) -> str | None:
        if not isinstance(raw, str):
            return None
        upper = raw.strip().upper()
        if upper in VALID_INTENTS:
            return upper
        return _LLM_INTENT_ALIASES.get(raw.strip().lower())

    def _ollama_chat(self, user: str, system: str) -> str:
        import ollama  # type: ignore[import-not-found]

        client = ollama.Client(host=LOCAL_LLM_HOST)
        resp = client.chat(
            model=LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": 0.2, "num_predict": 80},
        )
        return resp["message"]["content"]

    def _keyword_intent(self, transcript: str, det: DetectionFrame) -> IntentResult:
        for pattern, build in _KEYWORDS:
            m = pattern.search(transcript)
            if m:
                result = build(m)
                if result.intent == "WHATDOYOUSEE":
                    visible = sorted({b.cls for b in det.boxes})
                    result.reply = (
                        f"I see {', '.join(visible)}." if visible else "I don't see anything obvious."
                    )
                return result
        return IntentResult(intent="IDLE", reply="Sorry, I didn't catch that.")

    def _piper_speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            subprocess.run(
                ["piper", "--model", PIPER_VOICE_PATH, "--output_file", wav.name],
                input=text.encode(),
                check=True,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(["aplay", "-q", wav.name], check=False, stderr=subprocess.DEVNULL)


def make_user_turn(text: str) -> VoiceTurn:
    return VoiceTurn(role="user", text=text)


def make_assistant_turn(text: str) -> VoiceTurn:
    return VoiceTurn(role="assistant", text=text)


def time_voice_turn() -> dict[str, float]:
    return {"t0": perf_counter()}
