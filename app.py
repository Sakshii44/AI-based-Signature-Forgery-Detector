import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
# class SignatureCNN(nn.Module):
#     def __init__(self):
#         super(SignatureCNN,self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1,32,kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32,64,kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Flatten(),
#             nn.Linear(64*36*53,128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128,1),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         return self.model(x)

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("siamese_signature_model.pth",map_location=DEVICE))
model.eval()

# IMG_SIZE = (150,220)
transform = transforms.Compose([
    transforms.Resize((150,220)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])


st.title("AI SIGNATURE FORGERY DETECTOR")
st.write("Upload two signature images below to check if they match.")


img1 = st.file_uploader("Upload Reference Signature (Genuine)", type=["png", "jpg", "jpeg"], key="ref")
img2 = st.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"], key="test")

if img1 and img2:
    image1 = Image.open(img1).convert("L")
    image2 = Image.open(img2).convert("L")

    st.image([image1,image2],caption=["Reference","Test"],width=200)

    input1 = transform(image1).unsqueeze(0).to(DEVICE)
    input2 = transform(image2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output1, output2 = model(input1,input2)
        distance = F.pairwise_distance(output1,output2).item()

        st.subheader("Similarity score:")
        st.write(f"{distance:.4f}")

        if distance<0.5:
            st.success("Signatures Match(Likely Genuine)")
        else:
            st.error("Signatures do not match(Likely forged)")