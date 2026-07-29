import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Plot dilution script started")
print("Current working directory:", os.getcwd())


INPUT_FILE = "results/spot_detection/count_spots/spots_per_fov.csv"
OUTPUT_FOLDER = "results/spot_detection/dilution_curves"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("Output folder:", OUTPUT_FOLDER)
print("Exists:", os.path.exists(OUTPUT_FOLDER))

# Read the per-FOV results
df = pd.read_csv(INPUT_FILE)

# Remove automatically saved CSV index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Columns found:", df.columns.tolist())
print(df.head())


# Change these only if your CSV uses different column names
CONDITION_COLUMN = "sample"
COUNT_COLUMN = "spots_count"


if CONDITION_COLUMN not in df.columns:
    raise KeyError(
        f"Could not find '{CONDITION_COLUMN}'. "
        f"Available columns are: {df.columns.tolist()}"
    )

if COUNT_COLUMN not in df.columns:
    raise KeyError(
        f"Could not find '{COUNT_COLUMN}'. "
        f"Available columns are: {df.columns.tolist()}"
    )


def parse_condition(condition: str) -> pd.Series:
    """
    Convert labels such as:
        Ctrl Soak (1:1000)
        AD Triton (1:100)
    into group, fraction and dilution columns.
    """
    text = str(condition).strip()

    group_match = re.search(
    r"(?:^|[_\s])(Ctrl|Control|AD)(?=[_\s]|$)",
    text,
    re.IGNORECASE,
)
    dilution_match = re.search(r"1\s*:\s*(\d+)", text)

    if group_match:
        group_raw = group_match.group(1).lower()
        group = "Control" if group_raw in {"ctrl", "control"} else "AD"
    else:
        group = np.nan

    fraction_lookup = {
        "soak": "Soak",
        "hom": "Homogenate",
        "homogenate": "Homogenate",
        "triton": "Triton",
        "sark": "Sarkosyl",
        "sarkosyl": "Sarkosyl",
        "sds": "SDS",
    }

    fraction = np.nan
    lowered = text.lower()

    for keyword, full_name in fraction_lookup.items():
        if keyword in lowered:
            fraction = full_name
            break

    dilution = (
        int(dilution_match.group(1))
        if dilution_match
        else np.nan
    )

    return pd.Series(
        {
            "group": group,
            "fraction": fraction,
            "dilution": dilution,
        }
    )


parsed = df[CONDITION_COLUMN].apply(parse_condition)
df = pd.concat([df, parsed], axis=1)

# Ensure spot count is numeric
df[COUNT_COLUMN] = pd.to_numeric(df[COUNT_COLUMN], errors="coerce")

# Remove rows that could not be interpreted
df = df.dropna(
    subset=["group", "fraction", "dilution", COUNT_COLUMN]
)

df["dilution"] = df["dilution"].astype(int)


# Produce a summary table
summary = (
    df.groupby(["fraction", "group", "dilution"])[COUNT_COLUMN]
    .agg(
        n_fov="count",
        mean_spots="mean",
        median_spots="median",
        sd_spots="std",
        min_spots="min",
        max_spots="max",
    )
    .reset_index()
)

summary.to_csv(
    os.path.join(OUTPUT_FOLDER, "dilution_summary.csv"),
    index=False,
)

print("\nDilution summary:")
print(summary.to_string(index=False))


# Make one graph for each biochemical fraction
fractions = ["Soak", "Homogenate", "Triton", "Sarkosyl", "SDS"]

for fraction in fractions:
    fraction_df = df[df["fraction"] == fraction].copy()

    if fraction_df.empty:
        print(f"No data found for {fraction}; skipping.")
        continue

    fig, ax = plt.subplots(figsize=(7, 5))

    for group in ["Control", "AD"]:
        group_df = fraction_df[fraction_df["group"] == group]

        if group_df.empty:
            continue

        group_summary = (
            group_df.groupby("dilution")[COUNT_COLUMN]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("dilution")
        )

        # Individual FOV values with slight horizontal jitter
        rng = np.random.default_rng(42)

        for dilution, dilution_df in group_df.groupby("dilution"):
            jitter = rng.normal(
                loc=0,
                scale=0.025,
                size=len(dilution_df),
            )

            # Work in log10 space for evenly spaced dilution categories
            x_values = np.log10(dilution) + jitter

            ax.scatter(
                x_values,
                dilution_df[COUNT_COLUMN],
                alpha=0.45,
                label=None,
            )

        # Mean ± SD
        ax.errorbar(
            np.log10(group_summary["dilution"]),
            group_summary["mean"],
            yerr=group_summary["std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=4,
            label=group,
        )

    dilution_values = sorted(fraction_df["dilution"].unique())

    ax.set_xticks(np.log10(dilution_values))
    ax.set_xticklabels([f"1:{value}" for value in dilution_values])

    ax.set_xlabel("Sample dilution")
    ax.set_ylabel("Spots per field of view")
    ax.set_title(f"{fraction} dilution curve")
    ax.legend(title="Group")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{fraction.lower()}_dilution_curve.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")