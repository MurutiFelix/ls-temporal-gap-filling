# notebooks/extract_gap_months.py
"""
Reads the Landsat monthly image-count diagnostic CSV and extracts all
months where Total_L2_Images == 0, formatted as a JS array literal
ready to paste into the GEE AVHRR validation download script.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "Landsat_Monthly_Image_Counts_1995_2025_AllMissions_L1_L2_20CV.csv"

df = pd.read_csv(DATA_PATH)

gap_df = df[df["Total_L2_Images"] == 0][["Year", "Month"]].sort_values(["Year", "Month"])

print(f"Total gap months: {len(gap_df)}")
print(f"Years affected: {gap_df['Year'].min()}–{gap_df['Year'].max()}\n")

# Build JS array literal
js_lines = []
for _, row in gap_df.iterrows():
    js_lines.append(f"  {{y: {int(row['Year'])}, m: {int(row['Month'])}}}")

js_array = "var gapMonths = [\n" + ",\n".join(js_lines) + "\n];"

print(js_array)

# Also save to a text file for easy copy-paste
out_path = ROOT_DIR / "data" / "Processed" / "eda" / "gap_months_js.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(js_array)
print(f"\n[✓] Saved to: {out_path}")