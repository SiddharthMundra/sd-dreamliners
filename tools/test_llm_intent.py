"""Battery test for the Gemma intent classifier.

Runs a list of phrases through `voice.derive_intent` (LLM path, NOT keyword
fallback) and prints raw LLM output, normalized intent, target, reply,
and per-phrase latency.

Usage on the Pi:
    python3 -m tools.test_llm_intent
    python3 -m tools.test_llm_intent "find my keys" "what do you see"
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

os.environ.pop("KEYWORD_INTENT_ONLY", None)

from pi.config import LOCAL_LLM_MODEL
from pi.models import Detection, DetectionFrame
from pi.services.voice import VoicePipeline

DEFAULT_PHRASES = [
    "find a bottle",
    "find my red water bottle",
    "what do you see",
    "describe what is in front of me",
    "status",
    "how are you",
    "help me",
    "I fell down",
    "play some music",
    "what time is it",
]


def _sample_detections() -> DetectionFrame:
    return DetectionFrame(boxes=[
        Detection(cls="bottle", conf=0.82, x=0.42, y=0.55, w=0.08, h=0.18),
        Detection(cls="chair", conf=0.71, x=0.78, y=0.62, w=0.20, h=0.30),
        Detection(cls="person", conf=0.93, x=0.20, y=0.50, w=0.15, h=0.40),
    ])


def main() -> None:
    phrases = sys.argv[1:] or DEFAULT_PHRASES
    print(f"model: {LOCAL_LLM_MODEL}")
    print(f"phrases: {len(phrases)}\n")

    voice = VoicePipeline()
    detections = _sample_detections()

    print("warming model with one throwaway call ...")
    t0 = perf_counter()
    voice.derive_intent("hello", detections)
    print(f"  warmup {(perf_counter() - t0) * 1000:.0f}ms\n")

    width = max(len(p) for p in phrases)
    fails = 0
    for phrase in phrases:
        t0 = perf_counter()
        result = voice.derive_intent(phrase, detections)
        ms = (perf_counter() - t0) * 1000
        marker = "OK " if result.intent != "IDLE" or phrase.strip() == "" else "??"
        if result.intent == "IDLE" and phrase.strip():
            fails += 1
        print(
            f"  {marker} {phrase:<{width}}  {ms:6.0f}ms  "
            f"intent={result.intent:<13} target={result.target!s:<20} "
            f"reply={result.reply!r}"
        )

    print()
    print(f"unrecognized (fell to IDLE): {fails}/{len(phrases)}")


if __name__ == "__main__":
    main()
