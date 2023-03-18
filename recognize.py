import numpy as np
import imutils
import pickle
import cv2
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
conn = MongoClient("mongodb://localhost")
db = conn.NBAPlayerData
collection = db.playerData
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
image = cv2.imread("LeBron.jpg")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        for player in players.get_active_players():
            if name in player['last_name']:
                playergamelogs = playergamelog.PlayerGameLog(player['id']).get_dict()['resultSets'][0]
                headers = playergamelogs['headers']
                data = playergamelogs['rowSet']
                games = [game for game in data]
                for j in range(len(games)):
                    for i in range(len(headers)):
                        myplayer = {headers[i]:games[j][i]}
                        cursor = collection.find(myplayer)
                        data = [d for d in cursor] 
                        print(data)

print("Show Image...")
cv2.imshow("Image", image)
cv2.waitKey(0)