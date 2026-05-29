# ALAS Test Suite

Unit tests are organised to **mirror `src/`**, so the module under test is
unambiguous from a test's location. Demos, benchmarks, and model/camera-driven
inspection tools live separately under [`../eval/`](../eval), not here.

## Layout

```
tests/
├── conftest.py              # Puts src/ on sys.path; output_dir() helper.
├── fixtures/                # Committed input assets (images, …).
├── ai/                      # Tests for src/ai
│   └── test_preprocessing.py
├── navigation/
│   ├── local_planner/       # Tests for src/navigation/local_planner
│   │   └── test_vfh.py
│   └── sensors/             # Tests for src/navigation/sensors
│       └── test_gps.py      # Hardware-in-the-loop (needs a real GPS module)
├── hw_test/                 # Hardware bring-up placeholders
└── TTS_STT_test/            # Voice I/O test placeholders
```

## Conventions

- **Inputs** are read from `tests/fixtures/`.
- **Outputs/artifacts** are written under `outputs/tests/<module>/`
  (git-ignored), e.g. `outputs/tests/ai/preprocessed.png`.
- Tests are **standalone-runnable** scripts and also collectable by `pytest`.
- Each test's docstring states its inputs, outputs, and a `cd`-to-root run line.

## Running

From the repository root:

```
pytest tests/
python tests/ai/test_preprocessing.py
python tests/navigation/local_planner/test_vfh.py
```

`tests/navigation/sensors/test_gps.py` requires real GPS hardware + pyserial and
is intended to be run on the Jetson Nano.

## Main-loop integration

The end-to-end main loop is exercised by running the system in mock mode from
`src/` (see the project README):

```
python -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup
```
