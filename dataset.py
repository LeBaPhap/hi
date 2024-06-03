import os
import pickle
import mediapipe as mp
import cv2

def process_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def extract_hand_landmarks(image_rgb, hands):
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks
    return None

def normalize_landmarks(landmarks):
    x_ = [landmark.x for landmark in landmarks]
    y_ = [landmark.y for landmark in landmarks]
    min_x, min_y = min(x_), min(y_)
    normalized = [(landmark.x - min_x, landmark.y - min_y) for landmark in landmarks]
    return [coord for point in normalized for coord in point]  # Flatten list

def save_data(data, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

def main(data_dir, output_file):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    data = []
    labels = []
    
    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        for img_path in os.listdir(dir_path):
            img_rgb = process_image(os.path.join(dir_path, img_path))
            hand_landmarks = extract_hand_landmarks(img_rgb, hands)
            if hand_landmarks:
                for hand_landmark in hand_landmarks:
                    normalized_landmarks = normalize_landmarks(hand_landmark.landmark)
                    data.append(normalized_landmarks)
                    labels.append(dir_)

    save_data(data, labels, output_file)

if __name__ == "__main__":
    DATA_DIR = './data'
    OUTPUT_FILE = 'data.pickle'
    main(DATA_DIR, OUTPUT_FILE)
