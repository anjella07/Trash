import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR.parent))
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import io
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel


classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


class Trash(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((14, 14))  # ← КЛЮЧЕВО
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x



transform_data = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Trash()
model.load_state_dict(torch.load('model_pth', map_location=device))
model.to(device)
model.eval()

Trash_app = FastAPI()


@Trash_app.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='File not found')

        img = Image.open(io.BytesIO(data))
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            result = pred.argmax(dim=1).item()
        return {'class': classes[result]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(Trash_app, host='127.0.0.1', port=8080)
