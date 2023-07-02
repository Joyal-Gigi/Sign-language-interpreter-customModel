import pickle

import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

#configuring mediapipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

#mapping corresponding labels to letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
wordString=""
while True:

    data_aux = []
    x_ = []
    y_ = []
    #get realtime video
    ret, frame = cap.read()

    H, W, _ = frame.shape

    #color conversion for mediapipe (as mediapipe can only landmark rgb)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #getting hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:

        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x and y coordinates of hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Calculate the differences between x and y coordinates and the minimum values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        
        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)]) #initializing model to predict

        # Get the predicted character based on the label
        predicted_character = labels_dict[int(prediction[0])]#matching predicted label to find letter

        #add letter to string(to make words)
        #press 'A' to add letter to word and 'c' to clear letter from word
        if cv2.waitKey(1) == ord('a'):
            wordString = wordString + predicted_character
            print(wordString)
        if cv2.waitKey(1) == ord('c'):
            wordString = wordString[:-1]
            print(wordString)

        # Draw the bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1-50, y1-50), (x2+50, y2+50), (0, 0, 0), 4) #customising hud(rectangle and text)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, "Word: " + wordString, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    #press q to terminate
    if cv2.waitKey(1) == ord('q'):
        break


# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

