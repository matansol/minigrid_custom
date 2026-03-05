# Assessing Policy Updates — OSF Project

A lightweight, reproducible package for analyzing the **Assessing Policy Updates** user study and generating all paper-ready figures directly from a Jupyter notebook.

---

## 📦 What’s in this project?

```
.
├── dpu_ploting.ipynb              # Main notebook: loads the data and generates all plots
├── requirements.txt               # Pinned dependencies for reproducible runs
├── minigrid_recording.mp4  # A short demonstration of the experimental feedback - demonstraion loop, which was the main part of the experiment.
└── data/
    ├── dpu_results_users.csv      # Per-user aggregated results
    ├── dpu_results_choices.csv    # Per-choice (episode-level) results
    └── data_explanation.md        # Human-readable description of every column in the datasets
```
- **`data_explanation.md`** explains **every feature** in `dpu_results_users.csv` and `dpu_results_choices.csv` so you can match plot labels to the underlying variables with confidence.

- **`dpu_ploting.ipynb`** includes two helper functions:
  - `plot_barplot(...)` – a convenience wrapper to quickly create consistent bar plots used in the paper.
  - `mannwhitneyu_test(group_a, group_b, alternative="two-sided")` – runs a Mann–Whitney U (Wilcoxon rank-sum) test and prints tidy stats.

---

## ▶️ Quickstart

1. **Download all files** (the notebook, the `data/` dir with the two CSVs) into the **same folder** on your machine.
2. (Recommended) Create and activate a fresh virtual environment. Choose the commands that match your OS / shell:

- Windows — PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Windows — Command Prompt (cmd.exe):
```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

- macOS / Linux (bash, zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks script execution, either run a one-time bypass:

```powershell
powershell -ExecutionPolicy Bypass -NoProfile -Command ". .\.venv\Scripts\Activate.ps1"
```

or allow local scripts for your user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# then re-run: .\.venv\Scripts\Activate.ps1
```

3. Install dependencies and launch Jupyter (use the venv's Python to avoid launcher / executable path issues):

```powershell

# install pinned dependencies
python -m pip install -r requirements.txt

# start the notebook
jupyter notebook dpu_ploting.ipynb
# or
python -m notebook dpu_ploting.ipynb
```

4. **Run all cells** (Kernel → Run All). The notebook will:
   - Load the datasets from `./data/`,
   - Generate all the figures used in the paper,
   - Optionally save figures to disk (if the notebook includes a save path in the plotting calls).

> **Tip:** If you see a file-not-found error, confirm the relative paths match the structure above (i.e., the `data/` directory is a sibling of the notebook).

---

## 🧰 Requirements

Install exact dependencies from the provided file (recommended to run via the venv Python):

```bash
python -m pip install -r requirements.txt
```

The key libraries are:
- `pandas`, `numpy` — data handling
- `matplotlib` — plotting
- `scipy` — statistical tests (Mann–Whitney U)
- `seaborn` — optional styling
- `notebook`, `ipykernel` — Jupyter runtime

---

## 📊 Common tasks

Below are representative usage patterns you’ll find (or can adapt) in the notebook. The exact argument names might differ slightly—use the function signatures in the notebook as the source of truth.


---

## 📁 Data overview (at a glance)

- **`data/dpu_results_users.csv`** — one row per participant; includes demographics, behavioral outcomes (e.g., number of choices, accepts), the selected **final agent** and its **mean**, trust/attitudes, and Likert-scale responses about explanation quality.
- **`data/dpu_results_choices.csv`** — one row per decision episode; includes the **previous** and **updated** agent names and scores (mean & feedback), the user’s update decision, decision time, whether the choice was optimal (mean vs. feedback), and the final agent attributes repeated for convenience.

For the **complete** description of every column, see **`data_explanation.md`**.

---

Happy plotting!
