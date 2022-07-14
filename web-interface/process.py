from flask import Flask, render_template, Response, request
from PIL import Image
from keras.models import load_model
from time import sleep
from flask_socketio import SocketIO, emit
import time
import io
import base64,cv2
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import img_to_array
from engineio.payload import Payload
import os

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*' )

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index1.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

# @socketio.on('catch-frame')
# def catch_frame(data):
#     emit('response_back', data)  

global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]

face_classifier = cv2.CascadeClassifier(r'/home/vyom/MAIN/Summer Internship/ECA Using Socket/web-interface/haarcascade_frontalface_default.xml')
classifier = load_model(r'/home/vyom/MAIN/Summer Internship/ECA Using Socket/web-interface/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            labels.append(label)

            emotions_values = {'Anger': prediction[0],
                            'Disgust': prediction[1],
                            'Fear': prediction[2],
                            'Happiness': prediction[3],
                            'Neutral': prediction[4],
                            'Sadness': prediction[5],
                            'Surprise': prediction[6]
                            }

            emotion = max(emotions_values, key=lambda x: emotions_values[x])

            CI = 0

            if emotion=='Neutral':
                CI = emotions_values['Neutral']*0.9
            elif emotion=='Happiness':
                CI = emotions_values['Happiness']*0.6
            elif emotion=='Surprise':
                CI = emotions_values['Surprise']*0.6
            elif emotion=='Sadness':
                CI = emotions_values['Sadness']*0.3
            elif emotion=='Disgust':
                CI = emotions_values['Disgust']*0.2
            elif emotion=='Anger':
                CI = emotions_values['Anger']*0.25
            else:
                CI = emotions_values['Fear']*0.3

            if CI >= 0.2 and CI <= 1:
                engagement = 'Engaged'
            else:
                engagement = 'Not Engaged'

    # emit the frame back
    emit('response_back', {'emotion':emotion, 'engagement':engagement})
    
    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)),1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt+=1
    if cnt==30:
        fps_array=[fps]
        cnt=0
    
if __name__ == '__main__':
    socketio.run(app,port=9990 ,debug=True)
   

