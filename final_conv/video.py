import streamlit as st
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

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

def main():
    st.set_page_config(page_title="Real-time Emotion Detection", layout="wide")
    st.title("Real-time Emotion Detection")

    # Load the pre-trained model
    loaded_model = EmotionClassifier()
    loaded_model =torch.load('./model_conv.pt', map_location=torch.device('cpu'))
    loaded_model.eval()

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error opening webcam")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading frame from webcam")
            break

        # Face Detection
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        # Process the first detected face (if any)
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Coordinates of the first detected face

            # Extract the face region
            face_frame = frame[y:y + h, x:x + w]

            # Preprocess the face frame
            transformed_face = data_transform(face_frame)
            input_tensor = transformed_face.unsqueeze(0)

            # Make predictions
            with torch.no_grad():
                output = loaded_model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()

            # Display the predicted emotion on the frame
            emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"][predicted_label]
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'Emotion: {emotion}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame with emotion prediction(s)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()

if __name__ == "__main__":
    main()
