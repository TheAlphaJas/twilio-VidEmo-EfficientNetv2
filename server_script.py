from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import requests
from ml_script import predict_for_video
from requests.auth import HTTPBasicAuth
from twilio.rest import Client
import numpy as np

os.environ["TWILIO_ACCOUNT_SID"] = "YOUR SID"
os.environ["TWILIO_AUTH_TOKEN"] = "YOUR AUTH KEY"

account_sid = os.environ["TWILIO_ACCOUNT_SID"] 
auth_token = os.environ["TWILIO_AUTH_TOKEN"] 

client = Client(account_sid, auth_token)

def download_video(url, username, password):
    auth = HTTPBasicAuth(username, password)
    response = requests.get(url, auth=auth, stream=True)
    return response

def save_video(response, filename):
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

app = Flask(__name__)

def send_message(text, to_num):
    print(str(to_num))
    message = client.messages.create(
    body=f"{text}",
    from_="whatsapp:+14155238886",
    to=str(to_num),
    )

@app.route("/", methods=["GET", "POST"])
async def reply_whatsapp():
    print(request.values)
    sender_number = request.values.get('From')
    sender_name = request.values.get("ProfileName")
    try:
        num_media = int(request.values.get("NumMedia"))
    except (ValueError, TypeError):
        return "Invalid request: invalid or missing NumMedia parameter", 400
    response = MessagingResponse()
    if not num_media:
        msg = response.message(f"Hey {sender_name}! Welcome to VidEmo bot! Send me a short video and Ill try to guess your emotion in it! My response can be negative, neutral or positive")
        return str(response)
    else:
        if (request.values.get('MediaContentType0') != 'video/mp4'):
            send_message("Please record/send a video. Any other file is not a valid input.", sender_number)
        else:
            send_message("Thank you for your video. Analyzing the sentiment, please wait...",sender_number)
            media_url = request.values.get('MediaUrl0') 
            save_video(download_video(media_url,os.environ["TWILIO_ACCOUNT_SID"],os.environ["TWILIO_AUTH_TOKEN"]), "video.mp4")
            emotion = await predict_for_video("video.mp4")
            dict = {
                0:"Negative",    
                1:"Neutral",
                2:"Positive"
            }
            print(emotion)
            #Reducing threshold for positive and neutral due to bias of model towards negative
            if (emotion[2] >= 0.30):
                emotion = "Positive"
            elif (emotion[1] >= 0.30):
                emotion = "Neutral"
            else:
                emotion = dict[np.argmax(emotion)]
            print(emotion)
            send_message(f"The sentiment of the video is: {emotion}", sender_number)
    return ""


if __name__ == "__main__":
    app.run(debug=True)