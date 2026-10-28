from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_PATH = (
    SCRIPT_DIR
    / "data"
    / "extracted_data_scoping_review_ml_wave_propagation_seismology_revised.xlsx"
)
OUTPUT_DIRS = [SCRIPT_DIR / "figs", REPO_ROOT / "figures"]


def count_matches(series: pd.Series, pattern: str) -> int:
    return int(series.fillna("").str.contains(pattern, case=False, regex=True).sum())


def main() -> None:
    df = pd.read_excel(DATA_PATH)

    acoustic_count = count_matches(df["Equation"], r"Acoustic")
    elastic_count = count_matches(df["Equation"], r"\bElastic\b")
    helmholtz_count = count_matches(df["Equation"], r"Helmholtz")

    dimension_counts = {
        "1D": count_matches(df["Dimensionality"], r"\b1D\b"),
        "2D": count_matches(df["Dimensionality"], r"\b2D\b"),
        "3D": count_matches(df["Dimensionality"], r"\b3D\b"),
    }

    data_counts = {
        "Synthetic": int(df["Data"].eq("Synthetic only").sum()),
        "Experimental": int(df["Data"].eq("Real only").sum()),
        "Both": int(df["Data"].eq("Synthetic and real").sum()),
    }

    tensorflow_count = count_matches(df["ML framework"], r"TensorFlow")
    pytorch_count = count_matches(df["ML framework"], r"PyTorch")
    fd_count = count_matches(
        df["Standard method"], r"\bFD\b|Finite Difference"
    )
    sem_count = count_matches(
        df["Standard method"], r"\bSEM\b|Spectral Element"
    )

    print(f"Studies: {len(df)}")
    print(
        "Formulation counts:",
        {
            "Acoustic": acoustic_count,
            "Elastic": elastic_count,
            "Helmholtz": helmholtz_count,
        },
    )
    print("Dimensionality counts:", dimension_counts)
    print("Data counts:", data_counts)
    print(
        "Framework counts:",
        {"TensorFlow": tensorflow_count, "PyTorch": pytorch_count},
    )
    print("Standard method counts:", {"FD": fd_count, "SEM": sem_count})

    color_blue = "#b9cbe0ff"
    color_blue_2 = "#c7d0daff"
    color_gray = "#d6d6d683"

    fig = plt.figure(figsize=(6.5, 2.0))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig)
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.30, top=0.95, wspace=0.35
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(
        ["Acoustic", "Elastic", "Helmholtz"],
        [acoustic_count, elastic_count, helmholtz_count],
        color=color_blue,
    )
    ax1.set_title("Formulation", fontsize=8)
    ax1.set_ylabel("Number of publications", fontsize=8)
    ax1.tick_params(axis="x", rotation=25, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        list(dimension_counts),
        list(dimension_counts.values()),
        color=color_blue_2,
    )
    ax2.set_title("Dimensionality", fontsize=8)
    ax2.tick_params(axis="x", rotation=25, labelsize=7)
    ax2.tick_params(axis="y", labelsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(
        list(data_counts),
        list(data_counts.values()),
        color=color_gray,
    )
    ax3.set_title("Data type", fontsize=8)
    ax3.tick_params(axis="x", rotation=15, labelsize=7)
    ax3.tick_params(axis="y", labelsize=8)

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    color_ml = "#b9cbe0ff"
    color_standard = "#d6d6d683"
    fig2 = plt.figure(figsize=(2.8, 4.0))
    gs2 = GridSpec(2, 1, height_ratios=[1, 1], figure=fig2)
    fig2.subplots_adjust(
        left=0.18, right=0.97, bottom=0.10, top=0.95, hspace=0.7
    )

    ax4 = fig2.add_subplot(gs2[0, 0])
    ax4.bar(
        ["TensorFlow", "PyTorch"],
        [tensorflow_count, pytorch_count],
        color=color_ml,
        width=0.6,
    )
    ax4.set_title("Machine learning frameworks", fontsize=8)
    ax4.set_ylabel("Number of publications", fontsize=7)
    ax4.tick_params(axis="x", labelsize=7)
    ax4.tick_params(axis="y", labelsize=7)

    ax5 = fig2.add_subplot(gs2[1, 0])
    ax5.bar(
        ["Finite differences", "Spectral elements"],
        [fd_count, sem_count],
        color=color_standard,
        width=0.6,
    )
    ax5.set_title("Standard numerical methods", fontsize=8)
    ax5.set_ylabel("Number of publications", fontsize=7)
    ax5.tick_params(axis="x", labelsize=7)
    ax5.tick_params(axis="y", labelsize=7)

    for ax in (ax4, ax5):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "equation_dimensionality_data_type.svg",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / "equation_dimensionality_data_type.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            output_dir / "frameworks_methods.svg",
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            output_dir / "frameworks_methods.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)
    plt.close(fig2)


if __name__ == "__main__":
    main()
