import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- NASTAVENÍ ---
THRESHOLD_NM = 0.0  # Filtr na malé pohyby/šum
INPUT_MASK = "test_only/*.json*"  # Bere .json i .jsonl (pokud jsou validní)

# overriduje INPUT_MASK pokud je files not None
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
    f"{FILES_ROOT}/run71-data-feast-overnight/hysteresis_dataset_20260302_234218_clean_updated.jsonl",
    f"{FILES_ROOT}/run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
]


def load_data(filepath):
    """Nacte JSONL i JSON array/object soubor do Pandas DataFrame."""
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


def load_and_process_deltas(filepath):
    """
    Načte JSON/JSONL a spočítá delty.
    """
    try:
        df = load_data(filepath)
        if df.empty:
            return None

        # Ošetření, pokud by tam chyběly sloupce
        required_cols = ['iteration', 'experiment_name', 'x_target_abs', 'y_target_abs', 'x_actual_abs', 'y_actual_abs']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {filepath}: Missing columns")
            return None

        # 2. Výpočet Delta (Změna oproti předchozímu kroku)
        # Groupby zajistí, že nepočítáme rozdíl mezi koncem exp 1 a začátkem exp 2
        df['dx_target'] = df.groupby(['iteration', 'experiment_name'])['x_target_abs'].diff()
        df['dy_target'] = df.groupby(['iteration', 'experiment_name'])['y_target_abs'].diff()
        df['dx_actual'] = df.groupby(['iteration', 'experiment_name'])['x_actual_abs'].diff()
        df['dy_actual'] = df.groupby(['iteration', 'experiment_name'])['y_actual_abs'].diff()

        # Odstraníme NaN (první kroky v každém experimentu nemají předchůdce)
        df = df.dropna()

        return df

    except Exception as e:
        print(f"Chyba při čtení {filepath}: {e}")
        return None


def analyze_jumps(df, filename):
    print(f"\n--- ANALÝZA SKOKŮ: {filename} ---")

    # 1. X-AXIS Logic (Filtrujeme jen velké příkazy)
    big_moves_x = df[abs(df['dx_target']) > THRESHOLD_NM].copy()
    print(f"Počet skoků X > {THRESHOLD_NM} nm: {len(big_moves_x)}")

    if len(big_moves_x) > 0:
        corr_xx = big_moves_x['dx_target'].corr(big_moves_x['dx_actual'])
        corr_xy = big_moves_x['dx_target'].corr(big_moves_x['dy_actual'])  # Crosstalk

        print(f"  [X Command] -> Reakce X: {corr_xx:.4f} ", end="")
        if corr_xx > 0.9:
            print("✅ OK")
        elif corr_xx < -0.9:
            print("❌ OPAČNĚ (INVERTED)")
        else:
            print("⚠️  NEJASNÉ")

        print(f"  [X Command] -> Reakce Y (Crosstalk): {corr_xy:.4f}")

    # 2. Y-AXIS Logic
    big_moves_y = df[abs(df['dy_target']) > THRESHOLD_NM].copy()
    print(f"Počet skoků Y > {THRESHOLD_NM} nm: {len(big_moves_y)}")

    if len(big_moves_y) > 0:
        corr_yy = big_moves_y['dy_target'].corr(big_moves_y['dy_actual'])
        corr_yx = big_moves_y['dy_target'].corr(big_moves_y['dx_actual'])  # Crosstalk

        print(f"  [Y Command] -> Reakce Y: {corr_yy:.4f} ", end="")
        if corr_yy > 0.9:
            print("✅ OK")
        elif corr_yy < -0.9:
            print("❌ OPAČNĚ (INVERTED)")
        else:
            print("⚠️  NEJASNÉ")

        print(f"  [Y Command] -> Reakce X (Crosstalk): {corr_yx:.4f}")

    return big_moves_x, big_moves_y


def plot_vectors(big_moves_x, big_moves_y, title_suffix=""):
    """
    Vykreslí vektory změn.
    X-osa: Povel (Delta Target)
    Y-osa: Reakce (Delta Actual)
    """
    plt.figure(figsize=(12, 6))

    # Levý graf: Povel X
    plt.subplot(1, 2, 1)
    if len(big_moves_x) > 0:
        plt.scatter(big_moves_x['dx_target'], big_moves_x['dx_actual'], c='blue', label='Measured X', alpha=0.6)
        plt.scatter(big_moves_x['dx_target'], big_moves_x['dy_actual'], c='red', marker='x', label='Crosstalk Y',
                    alpha=0.4)
        plt.axhline(0, color='k', linewidth=1)
        plt.axvline(0, color='k', linewidth=1)
        plt.title(f"Reakce na povel X (> {THRESHOLD_NM}nm)")
        plt.xlabel("Povel dX [nm]")
        plt.ylabel("Naměřeno [nm]")
        plt.legend()
        plt.grid(True)
    else:
        plt.title("Žádné velké pohyby X")

    # Pravý graf: Povel Y
    plt.subplot(1, 2, 2)
    if len(big_moves_y) > 0:
        plt.scatter(big_moves_y['dy_target'], big_moves_y['dy_actual'], c='green', label='Measured Y', alpha=0.6)
        plt.scatter(big_moves_y['dy_target'], big_moves_y['dx_actual'], c='red', marker='x', label='Crosstalk X',
                    alpha=0.4)
        plt.axhline(0, color='k', linewidth=1)
        plt.axvline(0, color='k', linewidth=1)
        plt.title(f"Reakce na povel Y (> {THRESHOLD_NM}nm)")
        plt.xlabel("Povel dY [nm]")
        plt.ylabel("Naměřeno [nm]")
        plt.legend()
        plt.grid(True)
    else:
        plt.title("Žádné velké pohyby Y")

    plt.suptitle(f"Analýza vektorů: {title_suffix}")
    plt.tight_layout()
    plt.show()


def main():
    global FILES
    if not FILES:
        FILES = glob.glob(INPUT_MASK)
        if not FILES:
            print(f"Žádné soubory v {INPUT_MASK}")
            return

    for fpath in FILES:
        df = load_and_process_deltas(fpath)
        if df is None or len(df) == 0:
            continue

        bx, by = analyze_jumps(df, os.path.basename(fpath))

        if len(bx) > 0 or len(by) > 0:
            plot_vectors(bx, by, title_suffix=os.path.basename(fpath))
            # Odstraň break, pokud chceš projet všechny soubory
            # break


if __name__ == "__main__":
    main()
