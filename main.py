from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

# Inicializa la aplicación FastAPI
app = FastAPI()

# Cargar el modelo
MODEL_PATH = "modelo/custom_cnn_scripted.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()

except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Modelo no encontrado. Por favor, verifica la ruta.")

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensiona a 256x256
    transforms.Grayscale(),  # Convierte a escala de grises
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


@app.get("/")
async def root():
    return {"message": "¡Servidor funcionando correctamente!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"Nombre del archivo: {file.filename}")
    print(f"Content Type: {file.content_type}")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato de imagen no válido. Usa JPEG o PNG.")

    # Leer y procesar la imagen
    try:
        image = Image.open(BytesIO(await file.read())).convert("L")
        print(f"Dimensiones originales de la imagen (PIL): {image.size}")

        # Aplicar transformaciones
        image = transform(image)
        print(f"Dimensiones después de transformaciones (PyTorch Tensor): {image.shape}")

        # Agregar dimensión batch
        image = image.unsqueeze(0).to(device)
        print(f"Dimensiones finales del tensor (con batch): {image.shape}")

        # Verificar dimensiones
        if image.shape != (1, 1, 256, 256):
            raise HTTPException(status_code=400, detail=f"Dimensiones incorrectas: {image.shape}")

        # Realizar predicción
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Mapeo de etiquetas (ajusta según tu modelo)
        class_mapping = {0: "Normal", 1: "Neumonía"}
        prediction = class_mapping.get(predicted.item(), "Desconocido")
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
