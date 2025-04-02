FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/raw data/entrenamiento data/prueba models output temp_uploads static/uploads

# Puerto expuesto
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]