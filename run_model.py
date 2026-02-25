"""Run VIPER drift predictions from a CSV input and save a plot.

This script mirrors the workflow in run_model.ipynb, but exposes it as a
callable CLI so it can be used in pipelines and automation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def load_trained_model(model_path: Path, weights_path: Path, device: str) -> torch.nn.Module:
    """Load the serialized VIPER model architecture and trained weights."""
    if not hasattr(torch, "load"):
        raise RuntimeError(
            "The installed 'torch' package does not expose torch.load(). "
            "Install PyTorch and retry."
        )

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to equatorial latitudes and add circular feature encodings."""
    features = df[df["mlat"].between(-5, 5)].copy()

    features["mlt_sin"] = np.sin(features["mlt"] * (2.0 * np.pi / 24.0))
    features["mlt_cos"] = np.cos(features["mlt"] * (2.0 * np.pi / 24.0))
    features["doy_sin"] = np.sin((features["doy"] - 1) * (2.0 * np.pi / 365.0))
    features["doy_cos"] = np.cos((features["doy"] - 1) * (2.0 * np.pi / 365.0))
    features["lon_sin"] = np.sin(features["glon"] * (2.0 * np.pi / 360.0))
    features["lon_cos"] = np.cos(features["glon"] * (2.0 * np.pi / 360.0))

    return features


def normalize_features(df: pd.DataFrame, scaler_path: Path) -> pd.DataFrame:
    """Apply the training-time scaler to the inference dataframe."""
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)

    col_names = df.columns.to_list()
    normalized = scaler.transform(df)
    normalized_df = pd.DataFrame(normalized, columns=col_names)

    return normalized_df.drop(columns=["mlt", "mlat", "glon"])


def predict_with_uncertainty(
    model: torch.nn.Module,
    normalized_df: pd.DataFrame,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate model predictions and MAD uncertainty estimates."""
    model_input = torch.tensor(normalized_df.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(model_input).detach().numpy().reshape(-1)

        sample_predictions: list[np.ndarray] = []
        for _ in range(samples):
            mc_preds = model(model_input).detach().numpy().reshape(-1)
            sample_predictions.append(mc_preds)

    stacked = np.array(sample_predictions)
    median = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - median), axis=0)

    return predictions, mad


def load_test_date(
    raw_df: pd.DataFrame,
    model: torch.nn.Module,
    scaler_path: Path,
    samples: int = 500,
    global_error: float = 8.3,
) -> pd.DataFrame:
    """Run VIPER inference for a single-day dataframe and return tidy output."""
    filtered_df = raw_df[raw_df["mlat"].between(-5, 5)].copy()
    feature_df = prepare_features(raw_df)
    normalized_df = normalize_features(feature_df, scaler_path=scaler_path)

    predictions, mad = predict_with_uncertainty(model, normalized_df, samples=samples)

    filtered_df["vz_pred"] = predictions
    filtered_df["MAD"] = mad
    filtered_df["error"] = filtered_df["MAD"] + global_error

    output_df = filtered_df[["mlt", "mlat", "glon", "vz_pred", "error"]]
    return output_df[output_df["glon"] < 0]


def _pivot_sector(pivot_df: pd.DataFrame, value_name: str, lon_min: float, lon_max: float) -> pd.DataFrame:
    sector_df = pivot_df.stack().reset_index()
    sector_df.columns = ["glon", "mlt", value_name]
    return sector_df[sector_df["glon"].between(lon_min, lon_max)]


def plot_vz(pred_df: pd.DataFrame, date_title: str, output_path: Path) -> None:
    """Create and save the longitudinal sector plot used in the manuscript."""
    grouped = pred_df.groupby(["mlt", "glon"], as_index=False).mean(numeric_only=True)
    vz_pivot = grouped.pivot(index="glon", columns="mlt", values="vz_pred")
    err_pivot = grouped.pivot(index="glon", columns="mlt", values="error")

    df_vz_1 = _pivot_sector(vz_pivot, "vz", -170, -100)
    df_vz_2 = _pivot_sector(vz_pivot, "vz", -125, -70)
    df_vz_3 = _pivot_sector(vz_pivot, "vz", -70, -42)
    df_vz_4 = _pivot_sector(vz_pivot, "vz", -42, -5)

    df_er_1 = _pivot_sector(err_pivot, "error", -170, -100)
    df_er_2 = _pivot_sector(err_pivot, "error", -125, -70)
    df_er_3 = _pivot_sector(err_pivot, "error", -70, -42)
    df_er_4 = _pivot_sector(err_pivot, "error", -42, -5)

    if sns is not None:
        colors = sns.color_palette("pastel", 8).as_hex()
    else:
        colors = [plt.get_cmap("Set2")(i) for i in range(8)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), sharey=True)

    ax.errorbar(df_vz_1["mlt"], df_vz_1["vz"], color=colors[0])
    ax.fill_between(df_vz_1["mlt"], df_vz_1["vz"] - df_er_1["error"], df_vz_1["vz"] + df_er_1["error"], alpha=0.3, color=colors[0])

    ax.errorbar(df_vz_2["mlt"], df_vz_2["vz"], color=colors[1])
    ax.fill_between(df_vz_2["mlt"], df_vz_2["vz"] - df_er_2["error"], df_vz_2["vz"] + df_er_2["error"], alpha=0.3, color=colors[1])

    ax.errorbar(df_vz_3["mlt"], df_vz_3["vz"], color=colors[2])
    ax.fill_between(df_vz_3["mlt"], df_vz_3["vz"] - df_er_3["error"], df_vz_3["vz"] + df_er_3["error"], alpha=0.3, color=colors[2])

    ax.errorbar(df_vz_4["mlt"], df_vz_4["vz"], color=colors[3])
    ax.fill_between(df_vz_4["mlt"], df_vz_4["vz"] - df_er_4["error"], df_vz_4["vz"] + df_er_4["error"], alpha=0.3, color=colors[3])

    ax.legend(["-142°", "-97°", "-52°", "-7°"], loc="upper left", frameon=False)
    ax.set_title(date_title)
    ax.set_xlabel("MLT [hr]")
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_ylabel("Vz [m/s]")
    ax.axhline(0, color="k", linestyle="-", alpha=0.5, linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="x", direction="out")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VIPER plasma drift inference for one day.")
    parser.add_argument("--input", default="example_day.csv", help="Input CSV with prepared features.")
    parser.add_argument("--model", default="VIPER_model.pt", help="Path to serialized model architecture.")
    parser.add_argument("--weights", default="VIPER_weights.pt", help="Path to trained model weights.")
    parser.add_argument("--scaler", default="VIPER_scaler.pkl", help="Path to scaler pickle.")
    parser.add_argument("--output", default="example_day.png", help="Output plot path.")
    parser.add_argument("--save-predictions", default=None, help="Optional path to save predictions CSV.")
    parser.add_argument("--date-title", default="2014-03-16", help="Plot title (typically YYYY-MM-DD).")
    parser.add_argument("--samples", type=int, default=500, help="Monte Carlo samples for MAD estimation.")
    parser.add_argument("--global-error", type=float, default=8.3, help="Global error term in m/s.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = load_trained_model(Path(args.model), Path(args.weights), device=args.device)
    raw_df = pd.read_csv(args.input)

    pred_df = load_test_date(
        raw_df=raw_df,
        model=model,
        scaler_path=Path(args.scaler),
        samples=args.samples,
        global_error=args.global_error,
    )

    plot_vz(pred_df=pred_df, date_title=args.date_title, output_path=Path(args.output))

    if args.save_predictions:
        pred_df.to_csv(args.save_predictions, index=False)


if __name__ == "__main__":
    main()
