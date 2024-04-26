import os
import re
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Initialization
model_path = 'models/model_NN_MP_for_st.h5'
model = tf.keras.models.load_model(model_path)

category_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize sign sentence
sign_sentence = []


# Model Code
def video_frame_callback(frame):
    global sign_sentence
    image = frame.to_ndarray(format='bgr24')  # treat as cv2 image
    image = cv2.flip(image, 1)
    height, width = image.shape[:-1]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        single_hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in
                                 results.multi_hand_landmarks[0].landmark]

        # Saving landmarks for model input
        x = np.array(single_hand_landmarks).reshape(1, 63)

        mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(255, 0, 255), thickness=4, circle_radius=2),
                                  connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(20, 180, 90), thickness=2, circle_radius=2)
                                  )

        # interval of 3 is taken as coord z is included
        x_max = int(width * np.max(x[0, ::3]))
        x_min = int(width * np.min(x[0, ::3]))
        y_max = int(height * np.max(x[0, 1::3]))
        y_min = int(height * np.min(x[0, 1::3]))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        # Flipping X-axis landmarks for Left Hand
        if (results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x >
                results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP].x):
            x[:, ::3] = 1 - x[:, ::3]

        # Applying threshold
        if np.max(model.predict(x)) >= 0.5:
            y_pred_idx = np.argmax(model.predict(x))
            y_pred_text = category_names[y_pred_idx]
            cv2.putText(image, y_pred_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

            # Add recognized gesture to sign sentence
            if y_pred_text == 'space':
                sign_sentence.append(' ')
            elif y_pred_text == 'del':
                if sign_sentence:
                    sign_sentence.pop()
            else:
                sign_sentence.append(y_pred_text)
         # Update sign sentence placeholder
        sign_sentence_placeholder.text = ''.join(sign_sentence)
        #print(''.join(sign_sentence))


    return av.VideoFrame.from_ndarray(image, format='bgr24')

# Streamlit Code
st.header('ASL Gesture Recognition App', divider='green')

with st.container(border=True):
    st.write('''
    This app can detect the hand gestures of the American Sign Language. 
    Click on `START` to open the webcam. 
    ''')
st.write('#')

# Callback Thread
webcam_placeholder = st.empty()

with webcam_placeholder:
    ctx = webrtc_streamer(key='webcam',
                          video_frame_callback=video_frame_callback,
                          rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                          media_stream_constraints={"video": True, "audio": False})

st.write('#')


# Display the sign_sentence as a string
st.write('## Sign Sentence')
with st.container(height=100):
    sign_sentence_placeholder = st.empty()

st.write('#')
clear_button = st.button('Clear Text')

if clear_button:
    sign_sentence = []
    sign_sentence_placeholder.text = ''.join(sign_sentence)
