import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (150,220)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.ImageFolder('preprocessed_dataset/train',transform=transform)
val_dataset = datasets.ImageFolder('preprocessed_dataset/val',transform=transform)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle = True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

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
criterion  = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader,desc=f"Epoch{epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE),labels.to(DEVICE).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = (outputs>0.5).float()
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    acc = correct/total
    print(f"Epoch {epoch+1} - Loss: {train_loss/total:.4f} - Accuracy: {acc:.4f}")

torch.save(model.state_dict(),"signature_cnn_pytorch.pth")
print("Model trained and saved as 'signature_cnn_pytorch.pth'")