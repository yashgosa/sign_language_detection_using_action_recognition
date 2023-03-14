## Import depedencies

import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


## Keypoints using MP Holistic

mp_holistic = mp.solutions.holistic # Holistic Model
mp_drawing = mp.solutions.drawing_utils # Drawing Utilities
cap = cv2.VideoCapture(0)

def mp_detection(img, model):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Color Conversion from bgr to rgb
    img = cv2.flip(img, 1)                          # Fliping the frame
    img.flags.writeable = False                     # Image is no longer writeable
    results = model.process(img)                    # Make prediction
    img.flags.writeable = True                      # Image is now writable
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      # Color Conversion from rgb to bgr
    return img, results

def draw_landmarks(img, results):
    # face Connections
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness = 1, circle_radius= 1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness = 1, circle_radius= 1))
    # Pose Connections
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness = 2, circle_radius= 1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness = 2, circle_radius= 1))
    # hand Connections
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness = 2, circle_radius= 1),
                              mp_drawing.DrawingSpec(color=(121, 44, 121), thickness = 2, circle_radius= 1))
    # hand Connections
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness = 2, circle_radius= 1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness = 2, circle_radius= 1))

# set media pipe model

def cam_holistic():
    with mp_holistic.Holistic(min_detection_confidence= 0.5,
                              min_tracking_confidence= 0.5) as holistic:

        while cap.isOpened():

            # reading feed
            _, img = cap.read()

            # Make detections
            img, results = mp_detection(img, holistic)

            # Drawing the landmarks
            draw_landmarks(img, results)

            #Show to screen
            cv2.imshow("OpenCV feed", img)

            #Breaking Gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


## Extract Keypoint Values

# Whenener there are no landmarks detected we will return a zero array
# We want to flatten it so that we can feed it into LSTM Model

def extract_keypoints(results):

    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(132)

    # 21 landmarks each with x, y, z
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()\
        if results.right_hand_landmarks else np.zeros(21*3)

    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()\
        if results.face_landmarks else np.zeros(1404)

    return np.concatenate([pose, face, lh, rh])

## Setup Folders for Collection
DATA_PATH = os.path.join('HP_Data')                     #path for exported pata
actions = np.array(["hello", "thanks", "iloveyou"])     #Actions that we try to detect
no_sequences = 30                                       #No of videos
sequence_length = 30                                    #No of frames in each video

def create_dirs(DATA_PATH, actions, no_sequences, sequence_length):

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

## Collecting the data

def collect_data(actions, no_sequences, sequence_length):
# set media pipe model
    with mp_holistic.Holistic(min_detection_confidence= 0.5,
                              min_tracking_confidence= 0.5) as holistic:

        #Loop through the actions
        for action in actions:
            #Loop through throught the sequence
            for sequence in range(no_sequences):
                # Loop through each frame
                for frame_num in range(sequence_length):

                    # reading feed
                    _, img = cap.read()

                    # Make detections
                    img, results = mp_detection(img, holistic)

                    # Drawing the landmarks
                    draw_landmarks(img, results)

                    #Adding Text
                    if frame_num == 0:

                        cv2.putText(img, "STARTING COLLECTION", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, f"Collecting frames for {action} Video Number: {sequence}",
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("OpenCV feed", img)
                        cv2.waitKey(1000)
                    else:

                        cv2.putText(img, f"Collecting frames for {action} Video Number: {sequence}",
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("OpenCV feed", img)

                    # Extracting keypoints and saving them
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    #Breaking Gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()

# collect_data(actions, no_sequences, sequence_length)

## Preprocessing our data and creating the label features

def preprocess():

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = [] #will contain all the frame landmarks in the given video

            for frame_num in range(sequence_length):
                _ = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(_)

            sequences.append(window) # will contain the frames landmarks of every single video
            labels.append(label_map[action]) # appending the labels

    X = np.array(sequences)

preprocess()























