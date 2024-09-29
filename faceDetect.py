import  cv2 as cv
import os


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)


face_detector = cv.CascadeClassifier('F:/XLA/opencv/faceRecognition/haarcascade_frontalfac_default.xml')

faceID = input("\nNhap ID nhan dien khuon mat: ")

print("\n[Infor] Khoi tao camera ...")
count = 0

print("\n[Infor] An Q de bat dau ...")

while(True):
    _, img = cap.read()

    cv.imshow("image", img)

    if cv.waitKey(1) == ord("q") :
       break

while(True):

    _, img = cap.read()

    img = cv.flip(img, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1

        print(cv.imwrite("faceRecognition/dataset/User." + str(faceID) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w]))

    cv.imshow("image", img)

    k = cv.waitKey(100) & 0xff

    if k == 27 or count > 30 :
       break

print("\n[Infor] Thoat")
cap.release()
cv.destroyAllWindows()

