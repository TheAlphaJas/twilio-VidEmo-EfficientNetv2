import cv2
import numpy as np

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('./video_processing/haarcascade_frontalface_alt.xml')

def extract_image_arrays_with_faces(video_path, detection_frequency=10):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % detection_frequency == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in detected_faces:
                face = frame[y:y+h, x:x+w]
                faces.append(face)

    cap.release()
    cv2.destroyAllWindows() 
    return faces

def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count / fps
    cap.release()
    return video_length



def get_face_arrays(video_path):
    frame_rate = get_video_frame_rate(video_path)
    video_length = get_video_length(video_path)
    total_frames = video_length*frame_rate
    detection_frequency = int(max((total_frames/256),1))
    detected_faces = extract_image_arrays_with_faces(video_path, detection_frequency)
    return detected_faces

