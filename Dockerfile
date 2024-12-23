FROM python:3.11-slim

WORKDIR /app

# Copiar el contenido del proyecto
COPY . /app

# Instalar dependencias del proyecto
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exponer el puerto
EXPOSE 8080

# Comando para iniciar la API
CMD ["python", "main.py"]
