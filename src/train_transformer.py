# src/train_transformer.py
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Transformers para detección de misoginia (ES)")
    parser.add_argument("--model", type=str, default="dccuchile/beto-base-spanish-wwm-cased",
                        help="Nombre del modelo HF (p.ej. dccuchile/beto-base-spanish-wwm-cased, PlanTL-GOB-ES/roberta-base-bne, dccuchile/distilbert-base-spanish-uncased)")
    parser.add_argument("--small", action="store_true", help="Usar train_small/val_small/test_small")
    parser.add_argument("--epochs", type=float, default=3, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch por dispositivo (train)")
    parser.add_argument("--max-length", type=int, default=128, help="Longitud máx. de tokens")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (en épocas)")
    parser.add_argument("--art", type=str, default=None, help="Carpeta de salida; por defecto artifacts/<alias_modelo>")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    args = parser.parse_args()

  # === ALIASES (NUEVO): colocar inmediatamente después de parse_args ===
    aliases = {
    "BSC-LT/roberta-base-bne": "bertin-project/bertin-roberta-base-spanish",
    "PlanTL-GOB-ES/roberta-base-bne": "bertin-project/bertin-roberta-base-spanish",
    "roberta-base-bne": "bertin-project/bertin-roberta-base-spanish",
    }
    args.model = aliases.get(args.model, args.model)

    # Rutas y salida
    suffix = "_small" if args.small else ""
    DATA = Path("data")
    alias = args.model.split("/")[-1].replace("-", "_")
    ART = Path(args.art or f"artifacts/{alias}")
    ART.mkdir(parents=True, exist_ok=True)

    # Cargar CSVs
    train = pd.read_csv(DATA / f"train{suffix}.csv")
    val   = pd.read_csv(DATA / f"val{suffix}.csv")
    test  = pd.read_csv(DATA / f"test{suffix}.csv")
    for df in (train, val, test):
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok_batch(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    def to_hfds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.map(tok_batch, batched=True)
        drop_cols = [c for c in ["text", "__index_level_0__"] if c in ds.column_names]
        ds = ds.remove_columns(drop_cols)
        ds = ds.with_format("torch")
        return ds

    train_ds, val_ds, test_ds = map(to_hfds, (train, val, test))

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Métricas
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }

    # Entrenamiento
    torch.manual_seed(args.seed)
    use_fp16 = torch.cuda.is_available()

    args_tr = TrainingArguments(
        output_dir=str(ART / "trainer_out"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(2 * args.batch_size, 16),
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",   # 👈 CORREGIDO
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=args.seed,
        fp16=use_fp16,
        gradient_accumulation_steps=1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

    # Evaluación final en test
    metrics_test = trainer.evaluate(test_ds)
    print("== Métricas Test ==")
    print(metrics_test)

    # Predicciones test para reportes
    preds_logits = trainer.predict(test_ds).predictions
    y_pred = preds_logits.argmax(axis=-1)
    y_true = test["label"].to_numpy()

    # Guardar métricas + reporte + confusión
    with open(ART / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics_test.items()}, f, indent=2, ensure_ascii=False)

    rep_txt = classification_report(y_true, y_pred, zero_division=0)
    with open(ART / "transformer_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_txt)
    rep_json = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(ART / "transformer_report.json", "w", encoding="utf-8") as f:
        json.dump(rep_json, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No misógino", "Misógino"])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion - {alias}")
    fig.tight_layout()
    fig.savefig(ART / "transformer_confusion.png")
    plt.close(fig)

    # Guardar modelo final (pesos + tokenizer) en ART
    trainer.save_model(ART)
    tok.save_pretrained(ART)

    # Guardar parámetros usados (para reproducibilidad)
    with open(ART / "params.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "small": args.small,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "fp16": use_fp16
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ Checkpoint y reportes guardados en: {ART}")

if __name__ == "__main__":
    # Evita bloqueos en Windows por multiprocesamiento
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
