# 1) Crear el entorno
py -3.12 -m venv .venv

# 2) Activarlo 
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip

# 3) Instalar lo que necesitas
pip install -r requirements.txt

# 4)Ejecutar
python src/baseline.py
python src/train_transformer.py \
  --model dccuchile/distilbert-base-spanish-uncased \
  --epochs 1 \
  --batch-size 32 \
  --lr 3e-5 \
  --max-length 128 \
  --art artifacts/distilbeto_es \
  --seed 42
python src/train_transformer.py \
  --model PlanTL-GOB-ES/roberta-base-bne \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 128 \
  --art artifacts/roberta_es \
  --seed 42
