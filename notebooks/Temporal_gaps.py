# notebooks/Temporal_gaps.py
"""
Temporal Gap Analysis & Visualizations for Landsat Image Coverage (1995–2010).
Analyzes months with vs. without images and saves figures to data/Processed/eda/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# 1. Path Setup (Relative to Repository Root)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_PATH = ROOT_DIR / "data" / "Landsat_Monthly_Image_Count_20%QC_1995_2010.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "Processed" / "eda"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load & Prepare Data
df = pd.read_csv(DATA_PATH)

# Presence flag (1 = Images present, 0 = Missing/Zero)
df["Has_Images"] = (df["Total_Images"] > 0).astype(int)

# 3. Figure 1: Presence Heatmap (With vs. Without Images)
pivot_presence = df.pivot(index="Year", columns="Month", values="Has_Images")

plt.figure(figsize=(11, 7))

# Red (#e74c3c) = No Images | Green (#2ecc71) = Has Images
sns.heatmap(
    pivot_presence,
    cmap=["#e74c3c", "#2ecc71"],
    cbar=False,
    linewidths=1.5,
    linecolor="white",
    annot=False,
)

plt.title(
    "Landsat Data Coverage (1995–2010): Months With vs. Without Images",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)

# Custom Legend
red_patch = mpatches.Patch(color="#e74c3c", label="No Images (Count = 0)")
green_patch = mpatches.Patch(color="#2ecc71", label="Has Images (Count > 0)")
plt.legend(
    handles=[red_patch, green_patch],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
)

plt.tight_layout()
heatmap_path = OUTPUT_DIR / "landsat_coverage_presence_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[✓] Saved presence heatmap to: {heatmap_path}")

# 4. Figure 2: Quantitative Counts Heatmap (Zeroes Highlighted)
pivot_counts = df.pivot(index="Year", columns="Month", values="Total_Images")

plt.figure(figsize=(12, 8))

# Heatmap masking zeros
sns.heatmap(
    pivot_counts,
    cmap="YlGnBu",
    annot=True,
    fmt="g",
    cbar_kws={"label": "Total Images"},
    mask=(pivot_counts == 0),
)

# Background color for masked (missing/zero) cells
plt.gca().set_facecolor("#dcdde1")

plt.title(
    "Monthly Landsat Image Counts (Gray = Zero Images)",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)

plt.tight_layout()
counts_heatmap_path = OUTPUT_DIR / "landsat_monthly_counts_heatmap.png"
plt.savefig(counts_heatmap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[✓] Saved counts heatmap to: {counts_heatmap_path}")

# 5. Figure 3: Total Month Summary Breakdown
df["Status"] = df["Total_Images"].apply(
    lambda x: "With Images" if x > 0 else "No Images"
)
status_counts = df["Status"].value_counts()

plt.figure(figsize=(6, 4))
bars = plt.bar(
    status_counts.index, status_counts.values, color=["#2ecc71", "#e74c3c"]
)

plt.title("Total Month Coverage Summary (1995–2010)", fontweight="bold")
plt.ylabel("Number of Months")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{int(yval)}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
summary_bar_path = OUTPUT_DIR / "landsat_coverage_summary.png"
plt.savefig(summary_bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[✓] Saved summary chart to: {summary_bar_path}")

print("\nEDA processing complete.")