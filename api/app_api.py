import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI(title="Brain Tumor Detection API")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hna n-7adhro el-chargement mta3 el-modèles mte3ek (Densenet w Resnet)
# (Thabbet mel-architecture mta3 el-modèles mte3ek s7i7a kima el-kdim)
def load_model(model_path, model_type="densenet"):
    import torchvision.models as models
    if model_type == "densenet":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(num_ftrs, 4)
        )
    else:
        # Hna n-riglou el-ResNet50 b-Sequential zeda!
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(num_ftrs, 4) # <-- 4 classes kima el-DenseNet
        )
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Chargement initial des modèles
try:
    densenet_model = load_model("best_brain_tumor_densenet121_pytorch.pth", "densenet")
    resnet_model = load_model("best_brain_tumor_resnet50_pytorch.pth", "resnet")
    print("🚀 Modèles chargés avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")

# Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_choice: str = "densenet"):
    # Read Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Choose Model
    model = densenet_model if model_choice == "densenet" else resnet_model
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        # Idha 3andek probabilités t7eb t-khrajhom zeda tnejjem t-7otthom hna
        
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"] 
    prediction = class_names[preds.item()]
    
    return {"prediction": prediction, "status": "success"}