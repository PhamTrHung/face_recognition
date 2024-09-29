import cv2 as cv
import numpy as np
from PIL import Image
import os

path = 'dataset'

recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImageAndLables(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_Image = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_Image, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+h])
            ids.append(id)
    
    return faceSamples, ids

print("\n[Infor] Dang tranning du lieu ...")
faces, ids = getImageAndLables(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n[Infor] {0} Khuon mat duoc train. Thoat".format(len(np.unique(ids))))
