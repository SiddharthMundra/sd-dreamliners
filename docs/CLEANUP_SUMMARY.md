# Cleanup summary

Repository hygiene pass, StarkHacks 2026. No runtime behavior changed;
this is all layout + docs + ignore rules. The wire protocol, env var
names, bench scripts, and service code are untouched.

## Deleted

- All `.DS_Store` files under the tree (6 copies).
- All `__pycache__/` directories under the tree.
- `m5stack-starkhacks/.pio/` — PlatformIO build artifacts; rebuilt on
  demand with `pio run`.

## Archived (moved to `archive/`, gitignored)

These were duplicate or obsolete copies of files that now live under
`pi/`. They are kept locally for emergency revert but not tracked.

- `archive/yolov8n.pt` — 6.5 MB YOLO weights that were sitting at repo
  root. Should be pulled at setup time, not vendored.
- `archive/duplicate-root-py/main.py` — older snapshot of `pi/main.py`.
- `archive/duplicate-root-py/config.py` — older snapshot of
  `pi/config.py` (had stale `socket://localhost:5555` default and
  wrong baud rate).
- `archive/duplicate-root-py/models.py` — older snapshot of
  `pi/models.py`.
- `archive/duplicate-root-py/__init__.py` — empty root package marker
  left over from a pre-`pi/` layout.
- `archive/duplicate-root-py/services/` — older snapshot of
  `pi/services/` (missing narrator, yolo_ei, yolo_qaihub,
  pi_mic_recorder).
- `archive/duplicate-root-py/bench/` — older snapshot of
  `pi/bench/` missing `yolo_backend.py`.
- `archive/duplicate-root-py/e2e_smoke.py`, `fake_m5.py`,
  `preview_server.py`, `test_serial_bridge.py` — older root-level
  copies of files that now live in `tools/`. The `tools/` versions are
  newer (larger, more features).
- `archive/duplicate-root-py/pi_narrator_orphan.py` — `pi/narrator.py`
  that was shadowed by the canonical `pi/services/narrator.py`.

## Added

- `README.md` — rewritten around the on-device-LLM + ESP-tokenizer
  framing, with a complete env-var table and wire-protocol table.
- `docs/ARCHITECTURE.md` — prose walkthrough of the two-loop design,
  the serial contract, and the per-file responsibilities.
- `docs/CLEANUP_SUMMARY.md` — this file.
- `.env.example` — every `BELT_*` variable with a commented default.
- `.editorconfig` — Python 4-space, C/C++ 2-space, LF everywhere.
- `.clang-format` — LLVM base, 100-col, 2-indent for the M5 firmware.
- `pyproject.toml` — ruff config, pytest config, project metadata.
- `.gitignore` — expanded to cover `.pio/`, `archive/`, `*.pt`,
  `*.onnx`, `*.eim`, `*.dlc`, `*.tflite`, audio captures, and bench
  results. The previous `.gitignore` was 13 lines; the new one is
  organized by category.

## Behavior changes

None. This was a pure layout + docs pass. No env vars renamed, no
defaults flipped, no service code touched, no wire protocol changes.

## Not done (deferred)

Everything past Phase 8 of the cleanup brief. Specifically:

- Phase 4 (logging + bounded waits) — service code was intentionally
  not touched. `print()` audit and bare-except audit belong in a
  follow-up pass.
- Phase 6 (module docstrings) — most services already have top-of-file
  docstrings; a full audit wasn't done.
- Phase 7 (smoke tests) — no `tests/` directory has been added. The
  existing `tools/test_*` scripts still work.
- Phase 9 (M5 firmware tidy) — `src/main.cpp` is already compact and
  well-commented; no edits were made.
- Phase 10's final re-audit.

If the demo holds together with this pass, we can continue the
behavioral cleanup after the hackathon. If something breaks, revert
the archive move in one `mv archive/duplicate-root-py/* .` and we're
back where we started.
