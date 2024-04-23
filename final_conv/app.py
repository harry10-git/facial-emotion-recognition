import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch.nn as nn 
import streamlit as st

# Define the CNN model
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjusted input size after flattening
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the transformation for preprocessing (modify input size if needed)
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Adjust based on your model's input size
    transforms.Grayscale(num_output_channels=3),  # Ensure input has 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained model (replace with your model loading logic)
loaded_model = EmotionClassifier()
loaded_model = torch.load('./model_conv.pt', map_location='cpu')   # Assuming CPU for simplicity
loaded_model.eval()

# Face Detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        transformed_face = data_transform(face_frame)
        input_tensor = transformed_face.unsqueeze(0)

        with torch.no_grad():
            output = loaded_model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

        emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"][predicted_label]
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f'Emotion: {emotion}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def main():
    st.title("Real-time Emotion Detection")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Error opening webcam")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error reading frame from webcam")
            break

        frame = detect_emotion(frame)
        st.image(frame, channels="BGR", caption='Real-time Emotion Detection', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
