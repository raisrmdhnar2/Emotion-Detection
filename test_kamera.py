import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F


#10. Model CNN
class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(128 * 16 * 16, 256)
        self.fc2   = nn.Linear(256, 7)  # 7 class: emosi
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        # x = x.view(-1, 128 * 14 * 14)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnimalCNN().to(device)
model.load_state_dict(torch.load("c:/Users/ADMIN/Documents/Learn/Emotion Detection/best_model.pth", map_location=device))
model.eval()

# --- Kelas ---
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Transformasi Gambar (sama seperti saat training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),
                         (0.5))
])

# --- Buka Kamera ---
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Tidak dapat membuka kamera.")
    exit()

print("✅ Kamera terbuka. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Tidak dapat membaca frame.")
        break

    # Konversi BGR (OpenCV) ke RGB (PIL)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)


    # Preprocess
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]

    # Tampilkan hasil di frame
    cv2.putText(frame, f"Prediksi: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Animal Detection Live", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

