from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from typing import Self


class Modelo:
    def __init__(self: Self, model_path: str, device_: torch.device) -> None:
        self.device: torch.device = device_
        self.model: torch.jit._script.RecursiveScriptModule = self.load_model(model_path)
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def load_model(self: Self, model_path: str) -> torch.jit._script.RecursiveScriptModule:
        try:
            model: torch.jit._script.RecursiveScriptModule = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Modelo no encontrado. Por favor, verifica la ruta.")

    def predict(self, image: Image.Image) -> str:
        image: Image = self.transform(image)
        image: Image = image.unsqueeze(0).to(self.device)

        if image.shape != (1, 1, 256, 256):
            raise HTTPException(status_code=400, detail=f"Dimensiones incorrectas: {image.shape}")

        with torch.no_grad():
            outputs: torch.Tensor = self.model(image)
            _, predicted = torch.max(outputs, 1)

        class_mapping: dict[int, str] = {0: "Normal", 1: "Neumonía"}
        return class_mapping.get(predicted.item(), "Desconocido")


class PredictionAPI:
    def __init__(self: Self, app_: FastAPI, model_predictor_: Modelo) -> None:
        self.app: FastAPI = app_
        self.model_predictor: Modelo = model_predictor_
        self.setup_routes()

    def setup_routes(self: Self) -> any:
        @self.app.get("/")
        async def root() -> dict[str, str]:
            return {"message": "¡Servidor funcionando correctamente!"}

        @self.app.post("/predict/")
        async def predict(file: UploadFile = File(...)) -> JSONResponse:
            print(f"Nombre del archivo: {file.filename}")
            print(f"Content Type: {file.content_type}")

            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(status_code=400, detail="Formato de imagen no válido. Usa JPEG o PNG.")

            try:
                image: Image = Image.open(BytesIO(await file.read())).convert("L")
                print(f"Dimensiones originales de la imagen (PIL): {image.size}")

                prediction: str = self.model_predictor.predict(image)
                return JSONResponse(content={"prediction": prediction})

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


app: FastAPI = FastAPI()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = "modelo/custom_cnn_scripted.pt"

model_predictor: Modelo = Modelo(MODEL_PATH, device)
prediction_api: PredictionAPI = PredictionAPI(app, model_predictor)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)