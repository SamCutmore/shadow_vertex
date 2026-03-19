# shadow_vertex

Visualisation of high-dimensional linear programming using the shadow vertex algorithm.

## Running the visualisation

**Prerequisites**

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Rust toolchain (for building the `linprog_core` extension)
- On Linux, OpenGL/EGL libs for the viewer: `libgl1`, `libegl1` (e.g. `sudo apt install libgl1 libegl1`)

**One-time setup**

```bash
uv sync --extra dev
uv run maturin develop -C rust_engine
```

If you use your own virtualenv (e.g. `venv`) instead of uv’s default `.venv`, run `uv sync --extra dev --active` so dependencies install into the active environment; then use that env’s Python for `maturin develop` and `start.py`.

**Run**

```bash
uv run ./start.py --tesseract --dim 4
```

Other examples:

```bash
uv run ./start.py --kleeminty --dim 3 --solver simplex
uv run ./start.py --tesseract --dim 4 --solver shadow_vertex --sigma 0.01 --seed 7
```
