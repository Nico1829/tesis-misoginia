# data/download.py
from pathlib import Path
import os, sys, json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfFolder

# --- Config ---
DATA = Path("data")
DATA.mkdir(exist_ok=True)

# --- Token HF (dataset gated) ---
# Usa primero la variable de entorno HF_TOKEN; si no existe, intenta leer el token guardado por 'huggingface-cli login'
HF_TOKEN = os.getenv("HF_TOKEN", None) or HfFolder.get_token()

def load_raw():
    try:
        print("Descargando dataset de Hugging Face...")
        raw = load_dataset("manueltonneau/spanish-hate-speech-superset", token=HF_TOKEN)
        print("OK:", raw)
        return raw
    except Exception as e:
        msg = (
            "ERROR: No se pudo descargar el dataset (es 'gated').\n"
            "Soluciones:\n"
            "  1) Ejecuta en tu consola:  huggingface-cli login   (y pega tu token)\n"
            "     - Si no tienes el CLI:  python -m huggingface_hub login\n"
            "  2) O exporta la variable HF_TOKEN con tu token personal y vuelve a correr:\n"
            "     - Windows (cmd):   set HF_TOKEN=tu_token\n"
            "     - PowerShell:      $env:HF_TOKEN='tu_token'\n"
            "     - Linux/Mac:       export HF_TOKEN=tu_token\n"
        )
        print(msg)
        raise

raw = load_raw()

# -------- utilidades de selección de columnas --------
def pick_text_col(dset):
    cols = dset.column_names
    for c in ["text", "tweet", "content", "body", "sentence", "message"]:
        if c in cols:
            return c
    for c, feat in dset.features.items():
        if getattr(feat, "dtype", "") in ("string", "large_string"):
            return c
    return cols[0]

def pick_label_col(dset):
    cols = dset.column_names
    for c in ["label", "labels", "target", "y", "is_sexist", "misogyny"]:
        if c in cols:
            return c
    return None

txt_col = pick_text_col(raw["train"])
lab_col = pick_label_col(raw["train"])

def normalize_labels(series):
    """Devuelve serie 0/1 o categorías numeradas si hay >2 clases."""
    # intento directo a int
    try:
        arr = series.astype(int)
        # normaliza a 0/1 si son {0,1}
        uniq = sorted(pd.unique(arr))
        if set(uniq).issubset({0,1}):
            return arr.astype(int)
        # si son otros enteros, reindexa a 0..K-1
        mapping = {v:i for i,v in enumerate(uniq)}
        return arr.map(mapping).astype(int)
    except Exception:
        # texto → intenta mappings comunes
        low = series.astype(str).str.lower()
        common = [
            {"non-misogyny":0, "misogyny":1},
            {"non_misogynous":0, "misogynous":1},
            {"non-sexist":0, "sexist":1},
            {"no":0, "si":1},
            {"negativo":0, "positivo":1},
        ]
        vals = set(low.unique().tolist())
        for cand in common:
            if vals.issubset(set(cand.keys())):
                return low.map(cand).astype(int)
        # último recurso: ordenar alfabéticamente y enumerar
        uniq = sorted(vals)
        mapping = {v:i for i,v in enumerate(uniq)}
        return low.map(mapping).astype(int)

def to_df(dset):
    df = dset.to_pandas()
    df = df.rename(columns={txt_col: "text"})
    if lab_col and lab_col in df.columns:
        df["label"] = normalize_labels(df[lab_col])
    else:
        df["label"] = 0  # placeholder si no hay etiqueta
    df = df[["text", "label"]].dropna().drop_duplicates(subset=["text"])
    return df

# -------- preparar splits --------
dfs = {split: to_df(raw[split]) for split in raw.keys()}  # ej. 'train', 'test'

def safe_split(df, test_size, random_state=42):
    vc = df["label"].value_counts()
    use_strat = (vc.min() >= 2) and (len(vc) >= 2)
    return train_test_split(
        df, test_size=test_size,
        stratify=df["label"] if use_strat else None,
        random_state=random_state
    )

if "train" in dfs and "test" not in dfs:
    base = dfs["train"]
    train_df, test_df = safe_split(base, test_size=0.15)
    train_df, val_df  = safe_split(train_df, test_size=0.1765)
else:
    train_df = dfs.get("train", pd.concat(dfs.values(), ignore_index=True))
    val_df   = dfs.get("validation", None)
    test_df  = dfs.get("test", None)
    if val_df is None or test_df is None:
        base = train_df
        train_df, test_df = safe_split(base, test_size=0.15)
        train_df, val_df  = safe_split(train_df, test_size=0.1765)

# -------- guardar CSV completos --------
train_df.to_csv(DATA/"train.csv", index=False)
val_df.to_csv(DATA/"val.csv", index=False)
test_df.to_csv(DATA/"test.csv", index=False)

# -------- versiones small para prototipo rápido --------
def strat_sample(df, n_per_class):
    return df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(n_per_class, len(g)), random_state=42)
    )

mini_train = strat_sample(train_df, 2500)
mini_val   = strat_sample(val_df,   500)
mini_test  = strat_sample(test_df,  500)

mini_train.to_csv(DATA/"train_small.csv", index=False)
mini_val.to_csv(DATA/"val_small.csv", index=False)
mini_test.to_csv(DATA/"test_small.csv", index=False)

# -------- info útil --------
print("Tam. train/val/test:", len(train_df), len(val_df), len(test_df))
print("Balance train:\n", train_df["label"].value_counts(normalize=True).round(3))
print("✅ Listo. Revisa la carpeta /data")
