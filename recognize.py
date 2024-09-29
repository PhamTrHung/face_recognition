import cv2 as cv
import numpy as np
import os


recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath)

font = cv.FONT_HERSHEY_PLAIN

id = 0

names = ['Pham Trong Hung', 'Nguyen Thanh Nghi', 'Vu Kim Hoang', 'Vu Kim Hieu', 'Nguyen']
msvs = ['652360', '6653330', '680913', '6651210']

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

while(True):
    _, img = cap.read()
    img = cv.flip(img, 1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(int(minW), int(minH)))
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        masinhvien = ''

        if confidence > 40:
            masinhvien = msvs[id]
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "..."
            confidence = "  {0}%".format(round(100 - confidence))

        cv.putText(img, id, (x+5, y-25), font, 1, (255,255,0), 2)
        cv.putText(img, masinhvien, (x+5, y-5), font, 1, (255,255,0), 2)
        #cv.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 2)

    cv.imshow("Nhan dien khuon mat", img)


    if cv.waitKey(1) == ord("q"):
       break

print("\n[Infor] Thoat")
cap.release()
cv.destroyAllWindows()