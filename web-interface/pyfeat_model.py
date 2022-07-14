from feat import Detector
import os

def detect_emotion(img_path):
    detector = Detector()

    #img_path = os.path.join("images", "download.jpg")

    results = detector.detect_image(img_path)

    emotions_values = {'Anger': results['anger'][0],
                    'Disgust': results['disgust'][0],
                    'Fear': results['fear'][0],
                    'Happiness': results['happiness'][0],
                    'Sadness': results['sadness'][0],
                    'Surprise': results['surprise'][0],
                    'Neutral': results['neutral'][0]}

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

    if CI >= 0.5 and CI <= 1:
        return {'Emotion':emotion, 'Engagement level': 'Engaged'}
    else:
        return {'Emotion':emotion, 'Engagement level': 'Not Engaged'}
