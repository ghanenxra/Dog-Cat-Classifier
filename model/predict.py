import os
import torch
from torchvision import transforms, models
from torch import nn

# =====================
# CONFIG
# =====================
classes = ["Cat", "Dog"]
device = "cpu"

# =====================
# MODEL DEFINITION
# =====================a
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# =====================
# SAFE MODEL LOADING
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =====================
# TRANSFORM (EXPORTED)
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
