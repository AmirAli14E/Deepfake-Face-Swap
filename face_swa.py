# Line 1: Your face image path - Line 2: Target face image path
your_face_path = r"ff\me.jpg"
target_face_path = r"ff\ronaldoo.jpg"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import time


# Enhanced Face Swapper Architecture without facenet-pytorch dependency
class EnhancedFaceSwapper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.L1Loss()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Built-in OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            # Residual Blocks
            *[self.residual_block(256) for _ in range(6)],

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        return model

    def residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def train(self, epochs=1000):
        your_face = Image.open(your_face_path).convert('RGB')
        target_face = Image.open(target_face_path).convert('RGB')

        real_A = self.transform(your_face).unsqueeze(0).to(self.device)
        real_B = self.transform(target_face).unsqueeze(0).to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            fake_B = self.model(real_A)
            loss = self.criterion(fake_B, real_B)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    def swap_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                swapped_face = self.model(face_tensor).cpu()
                swapped_face = swapped_face.squeeze().permute(1, 2, 0).numpy()
                swapped_face = (swapped_face * 127.5 + 127.5).astype(np.uint8)
                swapped_face = cv2.resize(swapped_face, (w, h))

            frame[y:y+h, x:x+w] = swapped_face

        return frame

    def run(self):
        print("Starting model training...")
        start_time = time.time()
        self.train()
        print(f"Training completed in {time.time()-start_time:.2f} seconds")

        print("Starting real-time face swapping...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            swapped_frame = self.swap_face(frame)
            cv2.imshow('Enhanced Face Swap - Press Q to quit', swapped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the program
if __name__ == "__main__":
    if not os.path.exists(your_face_path):
        raise FileNotFoundError(f"File {your_face_path} not found")
    if not os.path.exists(target_face_path):
        raise FileNotFoundError(f"File {target_face_path} not found")

    try:
        swapper = EnhancedFaceSwapper()
        swapper.run()
    except Exception as e:
        print(f"Error: {str(e)}")