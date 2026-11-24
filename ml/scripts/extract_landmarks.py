import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258

# Maps Shriya's vocab.csv gloss_id -> possible folder names in INCLUDE dataset
INCLUDE_FOLDER_ALIASES = {
    "THANK-YOU": ["Thank you", "Thank_you", "THANK YOU", "thank you"],
}

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def normalize_landmarks(seq):
    normalized = seq.copy()
    for t in range(seq.shape[0]):
        frame = seq[t]
        l_shoulder = frame[44:47]
        r_shoulder = frame[48:51]

        if np.any(l_shoulder) and np.any(r_shoulder):
            center = (l_shoulder + r_shoulder) / 2.0
            scale = np.linalg.norm(l_shoulder - r_shoulder)
            if scale < 1e-5:
                scale = 1.0
        else:
            center = np.array([0.5, 0.5, 0.0])
            scale = 1.0

        for i in range(0, 132, 4):
            if np.any(frame[i:i+3]):
                normalized[t, i:i+3] = (frame[i:i+3] - center) / scale

        for i in range(132, 258, 3):
            if np.any(frame[i:i+3]):
                normalized[t, i:i+3] = (frame[i:i+3] - center) / scale

    return normalized

def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        frames.append(extract_keypoints(results))
    cap.release()

    if len(frames) == 0:
        return None

    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH).astype(int)
    sampled_seq = np.array([frames[i] for i in indices], dtype=np.float32)
    return normalize_landmarks(sampled_seq)

def find_sign_folder(raw_video_dir, gloss_id):
    candidates = INCLUDE_FOLDER_ALIASES.get(gloss_id, []) + [
        gloss_id, gloss_id.lower(), gloss_id.capitalize(), gloss_id.replace("-", " ").title()
    ]
    for name in candidates:
        full_path = os.path.join(raw_video_dir, name)
        if os.path.isdir(full_path):
            return full_path
    return None

def build_dataset(raw_video_dir, output_npy_dir, gloss_list):
    os.makedirs(output_npy_dir, exist_ok=True)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for gloss_id in gloss_list:
            sign_in_dir = find_sign_folder(raw_video_dir, gloss_id)
            if not sign_in_dir:
                print(f"[WARNING] Folder not found for {gloss_id} inside {raw_video_dir}")
                continue

            sign_out_dir = os.path.join(output_npy_dir, gloss_id)
            os.makedirs(sign_out_dir, exist_ok=True)

            videos = [f for f in os.listdir(sign_in_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            for vid in tqdm(videos, desc=f"Extracting {gloss_id}"):
                out_path = os.path.join(sign_out_dir, os.path.splitext(vid)[0] + ".npy")
                if os.path.exists(out_path):
                    continue
                seq = process_video(os.path.join(sign_in_dir, vid), holistic)
                if seq is not None:
                    np.save(out_path, seq)

if __name__ == "__main__":
    with open("ml/vocab_25.json", "r") as f:
        gloss_list = json.load(f)["classes"]

    build_dataset(
        raw_video_dir="ml/data/raw_include",
        output_npy_dir="ml/data/processed_npy",
        gloss_list=gloss_list
    )