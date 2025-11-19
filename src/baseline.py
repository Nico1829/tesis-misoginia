# src/baseline.py
import argparse, json, os, random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, RocCurveDisplay)
import joblib


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_splits(data_dir: Path, use_small: bool, text_col: str):
    suffix = "_small" if use_small else ""
    train = pd.read_csv(data_dir / f"train{suffix}.csv")
    val   = pd.read_csv(data_dir / f"val{suffix}.csv")
    test  = pd.read_csv(data_dir / f"test{suffix}.csv")

    # Normaliza tipos
    for df in (train, val, test):
        if text_col not in df.columns:
            raise ValueError(f"La columna '{text_col}' no existe. Columnas: {df.columns.tolist()}")
        df[text_col] = df[text_col].astype(str)
        if "label" not in df.columns:
            raise ValueError("Falta la columna 'label' en los CSV.")
        df["label"]  = df["label"].astype(int)

    # Entrenamos con train + val, probamos en test
    X_train = pd.concat([train[text_col], val[text_col]], ignore_index=True)
    y_train = pd.concat([train["label"],   val["label"]],   ignore_index=True)
    X_test  = test[text_col]
    y_test  = test["label"]
    return X_train, y_train, X_test, y_test


def build_pipeline(min_df: int, max_df: float, ngram_range, C: float):
    n1, n2 = ngram_range
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(n1, n2),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=C,
            solver="lbfgs"
        ))
    ])
    return pipe


@dataclass
class BaselineConfig:
    data: str = "data"
    art: str = "artifacts/baseline_es"
    reports: str = "reports"
    small: bool = False
    text_col: str = "text"       # o "text_clean"
    min_df: int = 3
    max_df: float = 0.9
    ngrams: tuple[int, int] = (1, 2)
    C: float = 1.0
    seed: int = 42


def run_baseline(cfg: BaselineConfig):
    set_seeds(cfg.seed)

    DATA = Path(cfg.data)
    ART  = Path(cfg.art);     ART.mkdir(parents=True, exist_ok=True)
    REPO = Path(cfg.reports); REPO.mkdir(parents=True, exist_ok=True)

    # 1) Cargar datos
    X_train, y_train, X_test, y_test = load_splits(DATA, cfg.small, cfg.text_col)

    # 2) Pipeline
    pipe = build_pipeline(cfg.min_df, cfg.max_df, cfg.ngrams, cfg.C)

    # 3) Entrenar
    pipe.fit(X_train, y_train)

    # 4) Evaluar
    y_pred = pipe.predict(X_test)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    rep_text = classification_report(y_test, y_pred, zero_division=0)
    rep_json = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"F1-macro (baseline): {f1m:.4f}")
    print(rep_text)

    # 5) Guardar artefactos y reportes
    joblib.dump(pipe, ART / "tfidf_lr.joblib")

    with open(ART / "params.json", "w", encoding="utf-8") as f:
        json.dump({
            "small": cfg.small,
            "text_col": cfg.text_col,
            "min_df": cfg.min_df,
            "max_df": cfg.max_df,
            "ngrams": list(cfg.ngrams),
            "C": cfg.C,
            "seed": cfg.seed
        }, f, indent=2, ensure_ascii=False)

    with open(REPO / "baseline_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_text)
    with open(REPO / "baseline_report.json", "w", encoding="utf-8") as f:
        json.dump(rep_json, f, indent=2, ensure_ascii=False)

    # 6) Figuras
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No misógino", "Misógino"])
    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Baseline TF-IDF + LR")
    fig.tight_layout()
    fig.savefig(REPO / "baseline_confusion.png")
    plt.close(fig)

    try:
        RocCurveDisplay.from_estimator(pipe, X_test, y_test)
        plt.title("Curva ROC — Baseline")
        plt.tight_layout()
        plt.savefig(REPO / "baseline_roc.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return {
        "f1_macro": f1m,
        "report_txt_path": str(REPO / "baseline_report.txt"),
        "report_json_path": str(REPO / "baseline_report.json"),
        "cm_png_path": str(REPO / "baseline_confusion.png"),
        "roc_png_path": str(REPO / "baseline_roc.png"),
        "model_path": str(ART / "tfidf_lr.joblib"),
        "params_path": str(ART / "params.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + Logistic Regression (misoginia español)")
    parser.add_argument("--data", type=str, default="data", help="Carpeta con train/val/test (.csv)")
    parser.add_argument("--art",  type=str, default="artifacts/baseline_es", help="Carpeta de artefactos")
    parser.add_argument("--reports", type=str, default="reports", help="Carpeta de reportes (PNG/TXT/JSON)")
    parser.add_argument("--small", action="store_true", help="Usar train_small/val_small/test_small")
    parser.add_argument("--text-col", type=str, default="text", help="Columna de texto (p.ej. text o text_clean)")
    parser.add_argument("--min_df", type=int, default=3, help="min_df del TF-IDF")
    parser.add_argument("--max_df", type=float, default=0.9, help="max_df del TF-IDF")
    parser.add_argument("--ngrams", type=str, default="1,2", help="ngram_range como '1,2' o '1,1'")
    parser.add_argument("--C", type=float, default=1.0, help="C de LogisticRegression")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    args = parser.parse_args()

    n1, n2 = map(int, args.ngrams.split(","))
    cfg = BaselineConfig(
        data=args.data, art=args.art, reports=args.reports, small=args.small,
        text_col=args.text_col, min_df=args.min_df, max_df=args.max_df,
        ngrams=(n1, n2), C=args.C, seed=args.seed
    )
    out = run_baseline(cfg)

    print(f"✅ Pipeline guardado en: {out['model_path']}")
    print(f"✅ Reportes/figuras en: {args.reports}")


if __name__ == "__main__":
    main()
