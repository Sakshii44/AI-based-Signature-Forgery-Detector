import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from siamese_dataset_loader import SiameseSignatureDataset
# from siamese_model import SiameseNetwork  # or paste the same model code here
from torchvision import transforms

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(64*36*53,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward_once(self,x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
    def forward(self,img1,img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1,output2
    
class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin

    def forward(self,output1,output2,label):
        distances = F.pairwise_distance(output1,output2)
        loss = torch.mean(
            (1-label)*torch.pow(distances,2)+label*torch.pow(torch.clamp(self.margin-distances,min=0.0),2)
        )
        return loss

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform
transform = transforms.Compose([
    transforms.Resize((150, 220)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
test_dataset = SiameseSignatureDataset("siamese_dataset", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("siamese_signature_model.pth", map_location=DEVICE))
model.eval()

# Evaluate
y_true = []
y_pred = []

THRESHOLD = 0.5  # You can try 0.4 or 0.6 to tune

with torch.no_grad():
    for img1, img2, labels in test_loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        labels = labels.to(DEVICE).float()

        out1, out2 = model(img1, img2)
        distances = F.pairwise_distance(out1, out2)

        preds = (distances < THRESHOLD).float()  # 1 if similar, else 0
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Forged", "Genuine"])

print(f"✅ Accuracy: {acc*100:.2f}%\n")
print("📊 Confusion Matrix:")
print(cm)
print("\n📝 Classification Report:")
print(report)
