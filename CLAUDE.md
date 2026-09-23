# gesture-remote — notes for Claude sessions

Windows background app: webcam → MediaPipe canned gestures → state machine → actions
(`keys`, `launch`, `url`, `script`), driven by `config.yaml`. Personal learning project,
100 % local. The approved phase-1 plan is `docs/plan-phase1.md` (French): it is the spec,
do not edit it.

## Hard rules

- **No network at runtime.** Only `tools/download_models.py` touches the network.
- **Frames stay in memory.** Never `imwrite`, `VideoWriter` or `imencode` in `src/`, `tools/`,
  `macros/` — `tests/test_privacy.py` fails on the bare words, comments included.
- **Never log images or landmarks.** Labels, scores and timings only.
- Code and comments in English, type hints everywhere, small modules, `logging` (no `print` in
  `src/`; macros may print, it goes to `logs/scripts/`).
- Write any file that contains a backslash (e.g. `shell:AppsFolder\`) with the Write tool, never
  through a Bash heredoc: the heredoc mangles `\`.
- Personal project: stays out of `~/work-kit`.

## Commands (source of truth — copy from here, not from memory)

Always call the venv interpreter; a bare `python` is the Microsoft Store alias on this machine.
From a worktree, use the **absolute** path of the main checkout's venv (worktrees have no venv).

```powershell
$PY = "C:\Users\adrie\gesture-remote\.venv\Scripts\python.exe"
& $PY -m pytest -q                 # whole suite; prints skipped tests (-ra in pyproject)
& $PY -m ruff check .
& $PY -m ruff format --check .
& $PY tools\download_models.py     # models + sample images, SHA-256 pinned
& $PY tools\check_setup.py         # models, sample images, raw class names, camera
& $PY tools\check_setup.py --no-camera
& $PY tools\check_setup.py --apps "Apple Music"   # Start menu name -> AppID
& $PY tools\check_setup.py --press playpause      # sends a REAL key press
& $PY -m gesture_remote --debug --dry-run         # from step 5 on
```

No `pip install` outside the main checkout; a missing dependency is requested from the admin.

## Architecture

```
Camera ─mirrored BGR─▶ IdleThrottle ─▶ Recognizer ─FrameObservation─▶ GestureEngine ─EngineEvent─▶ ActionDispatcher ─▶ keys | launch | url | script
(capture.py)                          (recognition.py)               (segments.py, engine.py)     (actions/)          └─▶ Feedback (beeps)
```

- `observation.py` — frozen dataclasses `HandObservation`, `FrameObservation` (`hands` is a
  **tuple**; `primary()` = highest score), `Handedness`, `NONE_LABEL = "none"`.
- `config.py` — pydantic models (`extra="forbid"`, frozen), discriminated union `ActionSpec` on
  `type`. Model validators are **context-free** only; labels, key names, Start-menu apps, file
  existence and "arrives in phase 3" messages belong to the loader.
- `features.py` — `normalize_landmarks` → `float32[63]` (aspect fix, wrist origin, /|p9| in 3D,
  mirror x for left hands; deliberately *not* rotation invariant: 👍 vs 👎). Used in phase 2.
- `segments.py` / `engine.py` — pure, time injected (`now`), see semantics below.
- `actions/` — `ActionDispatcher` (registry type → handler, one worker thread, never raises,
  `--dry-run`), `keys`, `launch` + `start_apps`, `url`, `script`.
- `app.py` — wiring + `Pipeline.run(stop: threading.Event)`; the main thread stays free for a
  phase-3 tray icon.

Phase seams: phase 2 swaps the recognizer behind the same `Recognizer` protocol; phase 3 adds
short/hold/combos on top of `segments.py` without changing it; phase 4 adds continuous detectors
on the same landmarks.

## Engine semantics (phase 1)

- A frame **votes** for its label if a hand is present, the label is not `none` and
  `score >= min_score(label)`.
- After `stable_frames(label)` identical consecutive votes a **segment** starts for that label
  (unless one is alive). A segment stays alive while its label was seen within `release_s`; a
  gesture returning before that continues its segment, even after another gesture in between.
- **Fire at onset only**, if mapped, armed and not in cooldown. Otherwise the segment is ignored
  for its whole life and the engine emits `Ignored(label, reason)` (logged, shown, no sound).
  Deliberately conservative: a deferred fire would trigger ✊ when the fist is laid on the desk
  right after another gesture.
- `repeat_while_held` (`keys` only): again after `repeat_delay_s`, then every
  `repeat_interval_s`, only on frames where the gesture is visible. Every fire restarts the
  global cooldown.
- 🤟 (`arm_gesture`) toggles armed/disarmed, **bypasses the cooldown** (so it can disarm right
  after a held 👍), one toggle per hold, and a toggle restarts the others' cooldown.
- (Re)start of the engine (launch, config reload) begins **in cooldown**.

## Decisions and traps (verified facts)

- mediapipe **1.0.1** on Python 3.13.7; only `mediapipe.tasks.python.vision` (no `mp.solutions`).
- **Class 0 is `None`, not `Unknown`** as the docs say. Read from
  `gesture_recognizer.task` → `hand_gesture_recognizer.task/canned_gesture_classifier.tflite/labels.txt`:
  `None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou`. Both
  `None` and `Unknown` map to `none`. Handedness labels: `Left`, `Right`.
- Sample images (IMAGE mode): `Thumb_Up` 0.73, `Thumb_Down` 0.77, `Victory` 0.91,
  `Pointing_Up` 0.82, 21 landmarks each (`check_setup.py`, 2026-09-23).
- **Camera: DirectShow opens it, 640×480, but only ~15 fps measured** (60 frames, mean brightness
  81/255, open 1.2 s; `check_setup.py`, 2026-09-23, cause not established). At 15 fps
  `stable_frames: 10` ≈ 0.67 s, not the 0.33 s assumed in the plan: to be tuned in step 6.
- OpenCV 5.0.0.93 (pulled by mediapipe): `CAP_DSHOW` and `CAP_MSMF` exist. Never add
  `opencv-python`. `VideoCapture.getBackendName()` raises once the capture is released.
- pyautogui 0.9.54: `press()` **silently ignores** unknown keys → validate with `isValidKey` (it
  exists). Set `FAILSAFE=False`, `PAUSE=0`.
- `Get-StartApps`: force UTF-8 output, decode `utf-8-sig`, accept a list **or** a single object
  (`ConvertTo-Json`). Verified: `Apple Music → AppleInc.AppleMusicWin_nzyj5cx40ttqa!App`, and
  `Intel® …` decodes to U+00AE. This Windows is in English: there is no `Paramètres` entry.
- The dispatcher lives for the whole session: a config reload replaces the mapping only, so the
  running-scripts registry and the Start-menu cache survive (otherwise a running script could be
  launched twice).
- Scripts: `sys.executable`, `CREATE_NO_WINDOW`, `cwd` = script folder, `PYTHONUTF8=1`,
  `PYTHONUNBUFFERED=1`, output to `logs/scripts/<name>.log`, one instance per script.
- `requirements.lock` must be written through `cmd /c "... > requirements.lock"` (PowerShell 5.1
  `>` writes UTF-16).
- `core.autocrlf=true` comes from the **system** gitconfig: working files are CRLF; normalise line
  endings before any text replacement.
- **Worktree import isolation is proven** (2026-09-23, at 6b19ef7, from `gesture-remote-vision`):
  with `pythonpath = ["src"]` the package is imported from the worktree; with
  `pytest -o pythonpath=` it falls back to main's `src/` through the editable `.pth` and
  `conftest.py` aborts with `UsageError` (exit 4). `-p no:python_path` is **not** a control:
  pytest 9 has no such plugin, so it disables nothing and the suite stays green.
- Never run `pip install` in the background of a session that may close: an interrupted install
  filled 508 `.py` files with NUL bytes.

## Conventions

- Labels are snake_case (`thumb_up`, `i_love_you`, `none`).
- New action type = model in `config.py` + handler in `actions/` + the exhaustiveness test.
- Every "tests green" claim cites the commit SHA and the number of skipped tests.
- Stage by explicit path; never `git add -A`, `git add .` or `commit -a`.

## Round 1 — ownership (parallel worktrees)

| Tree | Branch | Owner | Owns |
|---|---|---|---|
| `C:\Users\adrie\gesture-remote` | `main` | admin | everything below marked *frozen*, merges |
| `C:\Users\adrie\gesture-remote-vision` | `lot/vision` | session A | `features.py`, `recognition.py`, `capture.py`, `test_features.py`, `test_recognition.py`, `test_capture.py`, `test_mediapipe_integration.py` |
| `C:\Users\adrie\gesture-remote-decision` | `lot/decision` | session B | `config.py` (except public model names/fields), `config.yaml`, `segments.py`, `engine.py`, `test_config.py`, `test_segments.py`, `test_engine.py` |
| `C:\Users\adrie\gesture-remote-actions` | `lot/actions` | session C | `actions/*`, `feedback.py`, `test_actions.py`, `test_start_apps.py`, `test_feedback.py` |

*Frozen for the round (admin only, on request):* `pyproject.toml`, `tests/conftest.py`,
`tests/test_privacy.py`, `CLAUDE.md`, `README.md`, `.gitignore`, `requirements.lock`, `tools/*`,
`observation.py`, the model layer of `config.py`, `macros/example_hello.py`.

Exclusive resources, brokered by the admin: the **camera** (only the admin opens it), the
**keyboard** (no test sends a real key), no test launches a real app or URL — except the real
subprocess of the script test (in `tmp_path`) and a read-only `Get-StartApps` (`integration`).

## Progress

- [x] Step 0 — skeleton, gates (a) models, (b) camera, (c) Start menu — see SHA of the commit
      adding this file.
- [ ] Steps 1–4 — lots A, B, C.
- [ ] Step 5 — `recognition` wiring, `app.py`, `__main__.py`, `debug_view.py`, `logging_setup.py`.
- [ ] Step 6 — measurements and tuning with the user; phase 1 ✅ + SHA.
