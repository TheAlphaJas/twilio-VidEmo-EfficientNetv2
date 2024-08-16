<h1>Facial Expression Analysis with Fine-tuned EfficientNetV2-B1</h1>

This repository implements a facial expression analysis system using a fine-tuned EfficientNetV2-B1 model from Keras. It analyzes videos, classifying emotions into three categories: negative, neutral, and positive.

<b>Features:</b>

1. EfficientNetV2-B1 fine-tuned for facial expression recognition.
2. Video processing by splitting into frames and averaging predictions.
3. Backend Python scripts for machine learning tasks (requirements.py, ml_script.py).
4. Flask-based server script (server_script.py) for processing WhatsApp messages (Twilio integration).

<i>Note: The model currently exhibits a slight bias towards negative emotions. Thresholds are adjusted accordingly for positive and neutral classifications to reduce the bias. I am open to ideas, discussions and contributions on how I can improve further.</i>

--------------------------------------------------------------
Project Structure:

- requirements.py: Defines Python library dependencies for the ML side. flask[aysnc] is additionally required for the server.
- ml_script.py: Main script for loading the model, processing frames, and returning sentiment prediction.
- server_script.py: Flask application handling WhatsApp messages, forwarding requests to ml_script.py, and returning responses.
- image_model: This folder contains the images pre-processing script, the training notebook, some weights for trials, and the prediction script.
- video_processing: This folder contains the frame_extractor script (to extract each frame from a video), and the haarcascade xml file used for face cropping.

Getting Started:

- Clone the repository
- Install dependencies: Run pip install -r requirements.py.
- Run the backend: Execute python ml_script.py (for testing the model).
- Run the server (Optional): Execute python server_script.py to set up the Flask server (requires Twilio and ngrok configuration).

