# Hybrid CNN Inverter Simulation & Harmonic Analysis

An educational and research project for simulating a seven-level cascaded
H-bridge inverter, predicting reference waveforms with a CNN, and evaluating
carrier-phase optimization using the project simulator.

> **Scope:** This is an ideal software simulation. It is not hardware,
> MATLAB/Simulink, protection, or real-time control validation.

## What the project does

- Simulates a three-cell, seven-level inverter and its RL load.
- Displays waveforms, switching states, playback, and operating-point results
  in a Streamlit dashboard.
- Runs the existing CNN for **reference-target waveform prediction**. The CNN
  does not predict switching gates, control the inverter, or predict output THD.
- Compares PSO, POA, PFO, and hybrid POA-PFO carrier-phase search using
  simulator-evaluated candidates and electrical feasibility checks.
- Offers a separate CNN-guided candidate-ranking workflow. CNN scores are
  screening proxies; selected candidates are still evaluated by the simulator.
- Provides an optional local Ollama chat assistant that can request
  simulator-verified switching recommendations.

An optimizer run keeps the selected operating conditions fixed. A lower THD
is not guaranteed; use the recorded results and their measurement definitions
when evaluating performance.

## Important THD distinction

The project reports multiple THD measurements:

- **Event-based Parseval total voltage THD** is the primary objective for live
  voltage phase optimization and can include energy above harmonic 100.
- **Event-based H2-H100 THD** is a finite harmonic-band result.
- **Sampled FFT THD** is calculated from uniformly sampled waveforms; the
  dashboard uses a 20 kHz sample rate for its standard waveform.
- Load-current THD is calculated separately from the simulated RL-load current.

These metrics are not interchangeable. Dataset THD columns are dataset-reported
values, not measurements of CNN output or proof of inverter improvement.
See [THD conventions](./docs/thd_conventions.md) and the
[method comparison](./docs/THD_METHOD_COMPARISON.md).

## Requirements and project files

- Python 3.10 or newer
- Dependencies in [`requirements.txt`](./requirements.txt)
- The project dataset archive for dataset-backed features
- A compatible trained checkpoint for CNN inference and CNN-dependent features

The dataset archive and checkpoint are intentionally excluded from Git. Obtain
them separately and place them at these paths, or select/upload the appropriate
files in the dashboard:

```text
data/raw/Seven_Level_Inverter_Project.zip
colab_runs/baseline_20261120_080836/checkpoints/best.pt
```

The dataset contains 10,000 records split into 7,000 training, 1,500
validation, and 1,500 testing records. It uses 400 waveform samples per cycle,
20 kHz sampling, and a 50 Hz fundamental. Dataset and checkpoint files are
ignored by `.gitignore`; they are not downloaded automatically.

## Quick start (Windows PowerShell)

Run these commands from the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. For the full CNN workflow, provide a
compatible checkpoint and the dataset. Without a checkpoint, CNN-dependent
inference and simulation controls are unavailable; the dashboard reports the
missing requirement.

## Dataset validation and tests

With the dataset archive in place, validate and preprocess it without starting
training:

```powershell
python -c "from preprocessing.data_validation import validate_dataset; validate_dataset('data/raw/Seven_Level_Inverter_Project.zip', report_path='results/dataset_validation.json')"
python -c "from preprocessing.preprocess import preprocess_dataset; preprocess_dataset('data/raw/Seven_Level_Inverter_Project.zip', metadata_path='data/processed/preprocessing_metadata.json')"
```

Run the unit test suite:

```powershell
python -m unittest discover -s tests -v
```

Preprocessing aligns waveform tables by sample ID and fits normalization
statistics on the training split only. Test records are kept separate from
training.

## Training

Training is explicit; opening the dashboard or running the tests does not train
the CNN. For Google Colab, open [`notebooks/colab_train.ipynb`](./notebooks/colab_train.ipynb),
set `PROJECT_ROOT` and `DATASET_ZIP`, and run the notebook cells in order.

To deliberately start training locally:

```powershell
python -m training.train --config configs/config.yaml --dataset data/raw/Seven_Level_Inverter_Project.zip
```

Training creates experiment outputs and may write a new checkpoint. Do not use
this command if you only want to run inference with the existing checkpoint.

## Using the dashboard

The main **LIVE SIMULATION WORKSPACE** tab runs the selected operating point,
shows simulator-generated waveforms and metrics, and provides playback and
optimization controls. **RESEARCH WORKFLOW** contains the dataset overview,
CNN prediction, held-out test evaluation, and diagnostics.

The playback display replays a precomputed simulated cycle; it is not a
continuous signal or live hardware feed. The CNN waveform is a prediction of
the reference target and should not be read as measured inverter output.

For the optional local AI assistant, install and run Ollama separately, then
download the default model:

```powershell
winget install --exact --id Ollama.Ollama
ollama pull qwen2.5:3b
```

The assistant uses the local Ollama service. It is not a controller; gate
recommendations and their reported metrics come from the optimizer and
simulator.

## Results and documentation

- [Current project status](./docs/CURRENT_STATUS.md)
- [Scientific validation report](./SCIENTIFIC_VALIDATION_REPORT.md)
- [Simulator validation](./docs/SIMULATOR_VALIDATION.md)
- [Optimizer validation](./docs/optimizer_validation.md)
- [CNN held-out evaluation](./docs/CNN_GROUPED_TEST_EVALUATION.md)
- [CNN-guided optimization](./docs/cnn_guided_optimization.md)
- [Measured waveform validation](./docs/measured_waveform_validation.md)
- [THD surrogate screening](./docs/thd_surrogate_live_screening.md)
- [Final measured-results report](./docs/final_report.md)
- [Recorded results summary](./results/summary.md)
- [Project limitations](./PROJECT_LIMITATIONS.md)
- [Exhibition presentation guide](./EXHIBITION_PRESENTATION_GUIDE.md)

Recorded results describe specific operating points, budgets, and seeds; they
are not guarantees for other conditions. Review each report's methodology
before comparing numbers.

## Repository layout

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `multilevel/` | Inverter and RL-load simulation |
| `models/`, `training/` | CNN architecture, data loaders, and explicit training |
| `preprocessing/` | Dataset loading, validation, and normalization |
| `optimization/` | PSO, POA, PFO, hybrid search, and CNN-guided ranking |
| `evaluation/` | Live simulation, harmonic evaluation, and model metrics |
| `assistant/` | Local chat and simulator-verified recommendations |
| `scripts/`, `results/` | Experiment commands and recorded outputs |
| `docs/`, `tests/` | Technical documentation and automated tests |
