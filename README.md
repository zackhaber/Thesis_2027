# Thesis 2027: Pursuit-Evasion Benchmark

This project provides a small, reproducible benchmark for comparing cooperative pursuers in grid-based pursuit-evasion games. It includes difficulty-controlled level generation, pure pursuit/reach-avoid/maze modes, classical planner entry points, MARL training drivers, metrics, ASCII rendering, and notebook workflows.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py demo
python main.py benchmark
python main.py train --algorithm QMIX --episodes 100
python main.py notebook
```

Open the notebooks in `notebooks/` from top to bottom. The notebooks are deliberately thin: reusable experiment logic lives in `env/`, `planners/`, `marl.py`, `scenarios.py`, and `utils.py`.

## Project map

- `env/`: scenario generation and the shared environment API.
- `planners/`: CBS-style search, M*, Voronoi, and a time-expanded extension hook.
- `marl.py`: compatible training-driver interface for VDN, QMIX, QTRAN, and QPLEX baselines.
- `main.py`: simple command-line entry point for demos, benchmarks, training, and notebooks.
- `configs/`: example difficulty specifications.
- `notebooks/`: demonstrations, training, benchmarking, analysis, and UI preview.
- `ui.py`: dependency-free ASCII renderer suitable for early validation.

Running `python main.py` or `python main.py demo` opens the Tkinter simulation window. The pursuers and evader then move automatically; use Pause, Reset episode, or Close in the window.

The application flow is Main Menu → Game Mode → Algorithm → Difficulty → Simulation. Walls are directional barriers drawn as blue lines and still allow an agent to stand on their tile. Barrels are brown impassable cells. Maze scenarios generate walls only and never generate barrels.

The popup includes three selectable levels. Every agent moves at one fixed cell per turn:

| Level | Grid | Pursuers | Objects |
| --- | --- | --- | --- |
| 1 | 5 x 5 | 1 | None |
| 2 | 10 x 10 | 2 | Walls and barrels |
| 3 | 20 x 20 | 3 | Many walls and barrels |

## Research workflow

1. Establish environment and reward sanity in `01_environment_demo.ipynb`.
2. Add planner implementations and evaluate them on fixed scenarios.
3. Replace the lightweight MARL driver with the selected PyTorch/PyMARL implementation when the observation/action contract is stable.
4. Keep held-out scenarios for `04_benchmark_evaluation.ipynb` and report capture rate, time-to-capture, path efficiency, planning/training compute, and scaling.

The current planners and MARL drivers are intentionally small baselines, not claims of complete research-grade CBS, M*, Voronoi defense, or deep value-decomposition implementations. That separation keeps the project runnable while leaving a clean place to add the algorithms and citations required by the thesis.
