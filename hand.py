import pickle
import cv2
import mediapipe as mp
import numpy as np
import paho.mqtt.client as mqtt
import time

# Define the MQTT broker details
broker = "broker.emqx.io"
port = 1883
topic = "dacn3/esp32"
# Create an MQTT client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Connect to the broker
client.connect(broker, port, 60)

# Load the model
with open('./model.p', 'rb') as model_file:
    model_dict = pickle.load(model_file)
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up hand detection with MediaPipe
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Labels for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

# Set draw flag
draw = False
# Set the batch size and delay between batches (in seconds)
batch_size = 20  # Adjust as needed
batch_delay = 1  # Adjust as needed

# Initialize variables for batching
batch_data = []
last_batch_time = time.time()

while True:
    data_aux = []
    x_ = []
    y_ = []
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            if draw:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            # Normalize landmark coordinates
            min_x = min(x_)
            min_y = min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            # Predict the hand gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            
            # Add prediction to batch data
            batch_data.append(predicted_character)

            # Check if it's time to send a batch
            if len(batch_data) >= batch_size:
                # Publish batch message
                message = predicted_character
                client.publish(topic, message)

                # Reset batch data and update last batch time
                batch_data = []
                print(predicted_character)

            # Calculate bounding box coordinates
            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Draw bounding box and predicted character on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Disconnect from the broker
client.disconnect()
# Release resources
cap.release()
cv2.destroyAllWindows()
