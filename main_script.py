import numpy as np
from video_processing.face_extractor import get_face_arrays
from image_model.predictor import predict_from_frames
def predict_for_video(video_path):
    faces = get_face_arrays(video_path)
    outputs = predict_from_frames(faces)
    result = np.mean(outputs, axis=0)
    dict = {
        0:"negative",
        1:"neutral",
        2:"positive"
    }
    print("Emotion of video: ", dict[np.argmax(result)])

if __name__ == "__main__":
    print("Hello")
    video_path = input("Enter path of video: ")
    predict_for_video(video_path)