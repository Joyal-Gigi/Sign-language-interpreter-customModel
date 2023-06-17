import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

#configuring mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initializing the hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the image data is stored
DATA_DIR = './data'

# Lists to store the data and labels
data = []
labels = []

# Loop through each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image file in the current directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # List to store the data for the current image
        data_aux = []

        # Lists to store the x and y coordinates of landmarks
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Read the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the hands module
        results = hands.process(img_rgb)
        
        # Check if any hand landmarks were detected
        if results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each landmark in the hand
                for i in range(len(hand_landmarks.landmark)):     #extracting x and y attributes of landmark points
                    
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Append x and y coordinates to the respective lists
                    x_.append(x)
                    y_.append(y)
                # Calculate the difference between x and y coordinates and the minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            # Append the data and label to the respective lists
            data.append(data_aux)
            labels.append(dir_)

#creatie a pickle dataset with landmark coordinates for each class

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()