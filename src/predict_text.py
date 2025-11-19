# src/predict_text.py
import argparse, sys, json
from pathlib import Path
import joblib

LABELS = {0: "No Misoginia", 1: "Misoginia"}

def load_model(path: str):
    pipe = joblib.load(path)
    if not hasattr(pipe, "predict"):
        raise RuntimeError("El artefacto cargado no es un Pipeline sklearn válido.")
    return pipe

def predict_texts(pipe, textos):
    preds = pipe.predict(textos).tolist()
    out = []
    if hasattr(pipe, "predict_proba"):
        probas = pipe.predict_proba(textos).tolist()
        for t, y, p in zip(textos, preds, probas):
            out.append({
                "text": t,
                "pred": int(y),
                "label": LABELS.get(int(y), str(y)),
                "proba_no": float(p[0]),
                "proba_si": float(p[1]),
            })
    else:
        for t, y in zip(textos, preds):
            out.append({"text": t, "pred": int(y), "label": LABELS.get(int(y), str(y))})
    return out

def main():
    ap = argparse.ArgumentParser(description="Inferencia con baseline TF-IDF + LR")
    ap.add_argument("--model", type=str, default="artifacts/baseline_es/tfidf_lr.joblib", help="Ruta al joblib")
    ap.add_argument("--text", type=str, help="Texto único a clasificar")
    ap.add_argument("--file", type=str, help="Ruta a TXT con un texto por línea")
    args = ap.parse_args()

    pipe = load_model(args.model)

    textos = []
    if args.text:
        textos = [args.text]
    elif args.file:
        textos = [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        print("Provee --text o --file", file=sys.stderr); sys.exit(1)

    resultados = predict_texts(pipe, textos)
    print(json.dumps(resultados, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
