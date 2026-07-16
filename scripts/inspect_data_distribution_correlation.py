import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data_original"))

FILES = [
    f"{FILES_ROOT}/run33-complex/hysteresis_dataset_20251104_174024.jsonl",
    f"{FILES_ROOT}/run34-random678mag/hysteresis_dataset_20251114_102017.jsonl",
    f"{FILES_ROOT}/run35-random9-11-12/hysteresis_dataset_20251114_125908.jsonl",
    f"{FILES_ROOT}/run36-sawtooth-decreasing/hysteresis_dataset_20251118_170631.jsonl",
    f"{FILES_ROOT}/run37-sawtooth-complex-x/hysteresis_dataset_20251128_154751.jsonl",
    f"{FILES_ROOT}/run55-random-walk-20um/hysteresis_dataset_20260210_125219_updated.jsonl",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub0/confidence_0.8_no_axis_outliers_segments",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub1/confidence_0.8_no_axis_outliers_segments",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub2/confidence_0.8_no_axis_outliers_segments",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments",
    f"{FILES_ROOT}/run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
]


def expand_input_paths(paths):
    expanded = []
    for raw_path in paths:
        path = os.path.normpath(raw_path)
        if os.path.isdir(path):
            manifest_path = os.path.join(path, "manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8-sig") as f:
                        manifest = json.load(f)
                    segment_files = [
                        os.path.join(path, segment["file"])
                        for segment in manifest.get("segments", [])
                        if "file" in segment
                    ]
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    print(f"WARNING: Nepodařilo se načíst manifest {manifest_path}: {exc}")
                    segment_files = []
            else:
                segment_files = sorted(glob.glob(os.path.join(path, "*.jsonl")))

            existing = [file_path for file_path in segment_files if os.path.exists(file_path)]
            missing = [file_path for file_path in segment_files if not os.path.exists(file_path)]
            if missing:
                print(f"WARNING: {path} má v manifestu {len(missing)} chybějících segmentů.")
            if not existing:
                print(f"WARNING: Adresář {path} neobsahuje žádné JSONL segmenty.")
            expanded.extend(existing)
        else:
            expanded.append(path)
    return expanded

def load_data(filepath):
    """Nacte JSONL i JSON array soubor do Pandas DataFrame."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()

    if not content:
        return pd.DataFrame()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        return pd.DataFrame([parsed])
    except json.JSONDecodeError:
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return pd.DataFrame(data)


def analyze_axis_health(df, filename):
    """Provede statistickou analýzu korelací mezi Target a Actual"""

    # Vyfiltrujeme jen relevantní sloupce
    # Předpokládáme, že sloupce se jmenují 'x_target_abs', 'x_actual_abs' atd.
    required_cols = ['x_target_abs', 'y_target_abs', 'x_actual_abs', 'y_actual_abs']

    if not all(col in df.columns for col in required_cols):
        print(f"SKIP: Soubor {filename} neobsahuje potřebné sloupce.")
        return

    # Vytvoříme korelační matici
    # Zajímá nás vztah: Targety (řádky) vs Actuals (sloupce)

    tx = df['x_target_abs']
    ty = df['y_target_abs']
    ax = df['x_actual_abs']
    ay = df['y_actual_abs']

    # Pearsonova korelace
    corr_xx = tx.corr(ax)  # Target X vs Actual X (Mělo by být blízko 1.0)
    corr_xy = tx.corr(ay)  # Target X vs Actual Y (Mělo by být blízko 0.0)
    corr_yx = ty.corr(ax)  # Target Y vs Actual X (Mělo by být blízko 0.0)
    corr_yy = ty.corr(ay)  # Target Y vs Actual Y (Mělo by být blízko 1.0)

    print(f"\n--- ANALÝZA: {filename} ---")
    print(f"Vzorků: {len(df)}")

    # 1. Kontrola X osy
    print(f"X-Axis Health (Target X -> Actual X): {corr_xx:.4f} ", end="")
    if corr_xx > 0.9:
        print("✅ OK")
    elif corr_xx < -0.9:
        print("❌ INVERTED (Zrcadlově otočená)")
    else:
        print("⚠️  WEAK/NOISY (Nízká korelace)")

    # 2. Kontrola Y osy
    print(f"Y-Axis Health (Target Y -> Actual Y): {corr_yy:.4f} ", end="")
    if corr_yy > 0.9:
        print("✅ OK")
    elif corr_yy < -0.9:
        print("❌ INVERTED (Zrcadlově otočená)")
    else:
        print("⚠️  WEAK/NOISY (Nízká korelace)")

    # 3. Kontrola Cross-talku (Prohození os)
    print(f"Cross-Talk X->Y (Target X -> Actual Y): {corr_xy:.4f}")
    print(f"Cross-Talk Y->X (Target Y -> Actual X): {corr_yx:.4f}")

    if abs(corr_xy) > abs(corr_xx):
        print("\n🚨 KRITICKÁ CHYBA: Target X více koreluje s Y než s X! -> OSY JSOU PROHOZENÉ!")

    if abs(corr_yx) > abs(corr_yy):
        print("\n🚨 KRITICKÁ CHYBA: Target Y více koreluje s X než s Y! -> OSY JSOU PROHOZENÉ!")

    return tx, ty, ax, ay


def plot_diagnosis(tx, ty, ax, ay, filename):
    """Vykreslí scatter ploty pro vizuální kontrolu"""
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Axis Diagnosis: {filename}", fontsize=16)

    # 1. X vs X (Měla by být diagonála /)
    plt.subplot(2, 2, 1)
    plt.scatter(tx, ax, alpha=0.5, s=5, c='blue')
    plt.title(f"Target X vs Actual X\n(Expected: Diagonal /)")
    plt.xlabel("Target X")
    plt.ylabel("Actual X")
    plt.grid(True, alpha=0.3)

    # 2. X vs Y (Měl by být šum/mrak)
    plt.subplot(2, 2, 2)
    plt.scatter(tx, ay, alpha=0.5, s=5, c='red')
    plt.title(f"Target X vs Actual Y\n(Expected: Random Cloud)")
    plt.xlabel("Target X")
    plt.ylabel("Actual Y")
    plt.grid(True, alpha=0.3)

    # 3. Y vs X (Měl by být šum/mrak)
    plt.subplot(2, 2, 3)
    plt.scatter(ty, ax, alpha=0.5, s=5, c='red')
    plt.title(f"Target Y vs Actual X\n(Expected: Random Cloud)")
    plt.xlabel("Target Y")
    plt.ylabel("Actual X")
    plt.grid(True, alpha=0.3)

    # 4. Y vs Y (Měla by být diagonála /)
    plt.subplot(2, 2, 4)
    plt.scatter(ty, ay, alpha=0.5, s=5, c='green')
    plt.title(f"Target Y vs Actual Y\n(Expected: Diagonal /)")
    plt.xlabel("Target Y")
    plt.ylabel("Actual Y")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    # Nastav cestu k souborům (např. 'test_only' nebo složka s experimenty)
    # Zkusíme najít soubory s _MATH_FIX i bez něj
    search_path = "test_only/*.jsonl"

    files = expand_input_paths(FILES) if FILES else glob.glob(search_path)
    if not files:
        print(f"Žádné soubory nalezeny v {search_path}")
        return

    print(f"Nalezeno {len(files)} souborů.")

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"Skipping missing file {filepath}")
            continue

        filename = os.path.basename(filepath)
        df = load_data(filepath)

        if len(df) == 0:
            print(f"Skipping empty file {filename}")
            continue

        # Analýza čísel
        tx, ty, ax, ay = analyze_axis_health(df, filename)

        # Plotování (jen pro první soubor nebo odkomentuj pro všechny)
        # Doporučuji se podívat alespoň na jeden FIXED soubor
        plot_diagnosis(tx, ty, ax, ay, filename)


if __name__ == "__main__":
    main()
