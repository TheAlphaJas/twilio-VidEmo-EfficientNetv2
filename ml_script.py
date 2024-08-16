import numpy as np
from video_processing.face_extractor import get_face_arrays
from image_model.predictor import predict_from_frames
async def predict_for_video(video_path):
    faces = get_face_arrays(video_path)
    outputs = predict_from_frames(faces)
    result = np.mean(outputs, axis=0)
    dict = {
        0:"Negative",
        1:"Neutral",
        2:"Positive"
    }
    return result

if __name__ == "__main__":
    print("Hello! Welcome to VidEmo ML backend")
    video_path = input("Enter path of video: ")
    predict_for_video(video_path)