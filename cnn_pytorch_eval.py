import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

BATCH_SIZE = 32
IMG_SIZE = (150,220)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Correct for grayscale (1 channel)
])


test_dataset = datasets.ImageFolder('preprocessed_dataset/test',transform=transform)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

class SignatureCNN(nn.Module):
    def __init__(self):
        super(SignatureCNN,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*36*53,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)
    
model = SignatureCNN().to(DEVICE)
model.load_state_dict(torch.load('signature_cnn_pytorch.pth',map_location=DEVICE))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = (outputs>0.5).float().cpu().numpy().flatten()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

y_pred = [int(p) for p in y_pred]
y_true = [int(t) for t in y_true]

print("\n Confusion matrix:")
print(confusion_matrix(y_true,y_pred))

print("\n Classification report:")
print(classification_report(y_true,y_pred,target_names=['Forged','Genuine']))