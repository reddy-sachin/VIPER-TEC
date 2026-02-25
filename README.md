# VIPER-TEC

Prediction of equatorial vertical plasma drift from TEC using the VIPER neural network model.

Reference manuscript:
S. A. Reddy, X. Pi, C. Forsyth, A. Aruliah, A. Smith, *Earth and Space Science* 12(6), e2024EA004167.

<p align="center">
  <img src="01-jan-14.gif" width="500" height="400">
</p>

## What this repo contains

- `VIPER_model.pt`: serialized model architecture
- `VIPER_weights.pt`: trained weights
- `VIPER_scaler.pkl`: feature scaler used during training
- `example_day.csv`: prepared one-day input sample
- `run_model.ipynb`: original notebook workflow
- `run_model.py`: CLI version of the workflow for scripted runs
- `example_day.png`: example output figure

## Requirements

Python 3.9+ with:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `seaborn` (optional, used for color palette only)

Install manually (example):

```bash
pip install numpy pandas torch matplotlib seaborn
```

## Quick start

Run the model on the included sample input:

```bash
python run_model.py \
  --input example_day.csv \
  --model VIPER_model.pt \
  --weights VIPER_weights.pt \
  --scaler VIPER_scaler.pkl \
  --output example_day.png \
  --date-title 2014-03-16
```

This generates a sector-wise `Vz` plot similar to Figure 8 in the manuscript.

## CLI options

```bash
python run_model.py --help
```

Main options:

- `--input`: input CSV (prepared features)
- `--output`: path for output figure (`.png`, `.pdf`, etc.)
- `--save-predictions`: optional CSV export of predicted values and uncertainty
- `--samples`: Monte Carlo sample count for MAD uncertainty estimate (default `500`)
- `--global-error`: global error term in m/s (default `8.3`)
- `--device`: torch device (`cpu` or `cuda`)

## Input assumptions

`run_model.py` expects an already prepared CSV similar to `example_day.csv` with at least:

- `mlt`
- `mlat`
- `glon`
- `doy`
- any additional model input columns used in training

The script filters to equatorial latitudes (`-5 <= mlat <= 5`), applies the saved scaler, predicts `vz_pred`, computes uncertainty (`MAD` + global error), and plots negative longitudes only (`glon < 0`) to match the provided workflow.

## Notebook usage

If you prefer notebooks, open `run_model.ipynb` and execute cells top-to-bottom.
