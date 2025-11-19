# 1) Crear el entorno
py -3.12 -m venv .venv

# 2) Activarlo 
call .\.venv\Scripts\activate.bat

# 3) Actualizar pip dentro del venv (aquí SÍ tienes permisos)
python -m pip install --upgrade pip

# 4) Instalar lo que necesitas
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio