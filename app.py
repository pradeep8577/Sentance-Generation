import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st


# Initialization
model_path = 'models/sign_language_model.h5'
model = tf.keras.models.load_model(model_path)

category_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


# Streamlit Code
st.set_page_config(layout="wide")

st.header('Sign Language Detection', divider='green')

col1, col2 = st.columns(2)

with col1:
    webcam_placeholder = st.empty()

with col2:
    with st.container(height=300):
        text_output_placeholder = st.empty()

    st.write('#')
    clear_button = st.button('Clear Text')

    if clear_button:
        text_output = []
        text_output_placeholder.text(''.join(text_output))

text_output = []

st.write('#')


# Model Code
current_pred = None
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:-1]
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        single_hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in
                                 results.multi_hand_landmarks[0].landmark]

        # Saving landmarks for model input
        x = np.array(single_hand_landmarks).reshape(1, 63)

        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
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

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        # Flipping X-axis landmarks for Left Hand
        if (results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x >
                results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP].x):
            x[:, ::3] = 1 - x[:, ::3]

        # Applying thresholding
        if np.max(model.predict(x)) >= 0.5:

            y_pred_idx = np.argmax(model.predict(x))
            y_pred_text = category_names[y_pred_idx]
            cv2.putText(frame, y_pred_text, (x_min, y_min-5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

            # When new gesture is detected
            if current_pred != y_pred_text:
                current_pred = y_pred_text

                if y_pred_text == 'space':
                    text_output.append(' ')
                elif y_pred_text == 'del':
                    text_output.pop()
                else:
                    text_output.append(y_pred_text)

                text_output_placeholder.text(''.join(text_output))

    else:
        current_pred = None

    webcam_placeholder.image(frame, channels='BGR')#, use_column_width=True)

capture.release()
