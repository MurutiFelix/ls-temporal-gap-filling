# notebooks/Temporal_gaps.py
"""
Temporal Gap Analysis & Visualizations for Landsat Image Coverage (1995–2025).
Analyzes months with vs. without images (L2, science-ready) and saves figures
to data/Processed/eda/. Also flags months where L1 has scenes but L2 doesn't.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# 1. Path Setup (Relative to Repository Root)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_PATH = ROOT_DIR / "data" / "Landsat_Monthly_Image_Counts_1995_2025_AllMissions_L1_L2_20CV.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "Processed" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load & Prepare Data
df = pd.read_csv(DATA_PATH)

# Presence flag based on L2 (science-ready) counts - what your export pipeline uses
df["Has_Images"] = (df["Total_L2_Images"] > 0).astype(int)

# Flag: L1 has scenes but L2 doesn't - scene existed but didn't survive SR/ST processing
df["L1_Only_Gap"] = ((df["Total_L1_Images"] > 0) & (df["Total_L2_Images"] == 0)).astype(int)

# 3. Figure 1: Presence Heatmap (With vs. Without L2 Images)
pivot_presence = df.pivot(index="Year", columns="Month", values="Has_Images")

plt.figure(figsize=(11, 7))
sns.heatmap(
    pivot_presence,
    cmap=["#e74c3c", "#2ecc71"],
    cbar=False,
    linewidths=1.5,
    linecolor="white",
    annot=False,
)

plt.title(
    "Landsat Data Coverage (1995–2025): Months With vs. Without L2 Images",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)

red_patch = mpatches.Patch(color="#e74c3c", label="No L2 Images (Count = 0)")
green_patch = mpatches.Patch(color="#2ecc71", label="Has L2 Images (Count > 0)")
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

# 4. Figure 2: Quantitative Counts Heatmap (L2, Zeroes Highlighted)
pivot_counts = df.pivot(index="Year", columns="Month", values="Total_L2_Images")

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_counts,
    cmap="YlGnBu",
    annot=True,
    fmt="g",
    cbar_kws={"label": "Total L2 Images"},
    mask=(pivot_counts == 0),
)
plt.gca().set_facecolor("#dcdde1")

plt.title(
    "Monthly Landsat L2 Image Counts (Gray = Zero Images)",
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

# 5. Figure 3: Total Month Summary Breakdown (L2)
df["Status"] = df["Total_L2_Images"].apply(
    lambda x: "With Images" if x > 0 else "No Images"
)
status_counts = df["Status"].value_counts()

plt.figure(figsize=(6, 4))
bars = plt.bar(
    status_counts.index, status_counts.values,
    color=["#2ecc71" if s == "With Images" else "#e74c3c" for s in status_counts.index]
)

plt.title("Total Month Coverage Summary, L2 (1995–2025)", fontweight="bold")
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

# 6. Figure 4: L1-only gap heatmap (scenes existed at L1 but not L2)
pivot_l1gap = df.pivot(index="Year", columns="Month", values="L1_Only_Gap")

plt.figure(figsize=(11, 7))
sns.heatmap(
    pivot_l1gap,
    cmap=["#f0f0f0", "#e67e22"],
    cbar=False,
    linewidths=1.5,
    linecolor="white",
    annot=False,
)

plt.title(
    "Months Where L1 Scenes Exist But L2 SR/ST Doesn't",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)

none_patch = mpatches.Patch(color="#f0f0f0", label="No Gap")
gap_patch = mpatches.Patch(color="#e67e22", label="L1 Present, L2 Missing")
plt.legend(
    handles=[none_patch, gap_patch],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
)

plt.tight_layout()
l1gap_path = OUTPUT_DIR / "landsat_l1_only_gap_heatmap.png"
plt.savefig(l1gap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[✓] Saved L1-only gap heatmap to: {l1gap_path}")

print("\nEDA processing complete.")