import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILES_ROOT = "../data_original"

FILES = [
    f"{FILES_ROOT}/run33-complex/hysteresis_dataset_20251104_174024.jsonl",
    f"{FILES_ROOT}/run34-random678mag/hysteresis_dataset_20251114_102017.jsonl",
    f"{FILES_ROOT}/run35-random9-11-12/hysteresis_dataset_20251114_125908.jsonl",
    f"{FILES_ROOT}/run36-sawtooth-decreasing/hysteresis_dataset_20251118_170631.jsonl",
    f"{FILES_ROOT}/run37-sawtooth-complex-x/hysteresis_dataset_20251128_154751.jsonl",
    f"{FILES_ROOT}/run55-random-walk-20um/hysteresis_dataset_20260210_125219_updated.jsonl",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub0/hysteresis_dataset_reconstructed.jsonl",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub1/hysteresis_dataset_reconstructed.jsonl",
    f"{FILES_ROOT}/run70-data-feast-overnight-sub2/hysteresis_dataset_reconstructed.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part001_steps0000-0055.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part002_steps0057-0093.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part003_steps0095-0177.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part004_steps0179-0183.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part005_steps0185-0202.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part006_steps0204-0327.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part007_steps0329-0342.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part008_steps0344-0354.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part009_steps0356-0372.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part010_steps0374-0375.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part011_steps0377-0432.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part012_steps0434-0457.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part013_steps0459-0620.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part014_steps0622-0682.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part015_steps0684-0755.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part016_steps0757-0797.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part017_steps0799-0823.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part018_steps0825-0906.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part019_steps0908-0921.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part020_steps0923-1057.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part021_steps1059-1076.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part022_steps1078-1112.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part023_steps1114-1230.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part024_steps1232-1234.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part025_steps1236-1278.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part026_steps1280-1314.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part027_steps1316-1318.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part028_steps1320-1334.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part029_steps1336-1414.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part030_steps1416-1431.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part031_steps1433-1441.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part032_steps1443-1452.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part033_steps1454-1454.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part034_steps1456-1461.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part035_steps1463-1504.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part036_steps1506-1541.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part037_steps1543-1687.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part038_steps1689-1714.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part039_steps1716-1775.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part040_steps1777-1780.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part041_steps1782-1870.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part042_steps1872-1904.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part043_steps1906-1983.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part044_steps1985-2108.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part045_steps2110-2152.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part046_steps2154-2162.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part047_steps2164-2353.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part048_steps2355-2392.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part049_steps2394-2404.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part050_steps2406-2491.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part051_steps2493-2531.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part052_steps2533-2566.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part053_steps2568-2607.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part054_steps2609-2621.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part055_steps2623-2631.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part056_steps2633-2671.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part057_steps2673-2688.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part058_steps2690-2762.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part059_steps2764-2820.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part060_steps2822-2904.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part061_steps2906-2974.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part062_steps2976-2996.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part063_steps2998-3031.jsonl",
    f"{FILES_ROOT}/run71-data-feast-overnight/confidence_0.7_segments/hysteresis_dataset_20260302_234218_clean_updated_part064_steps3033-3035.jsonl",
    f"{FILES_ROOT}/run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
]

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

    files = FILES if FILES else glob.glob(search_path)
    if not files:
        print(f"Žádné soubory nalezeny v {search_path}")
        return

    print(f"Nalezeno {len(files)} souborů.")

    for filepath in files:
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