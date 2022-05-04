import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for j in images:
        j = cv2.cvtColor(j,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(j)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, j = cap.read()
    imgS = cv2.resize(j,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(j,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(j,(x1,y2-20),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(j,name,(x1+3,y2-3),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',j)
    cv2.waitKey(1)




#imgElon = face_recognition.load_image_file('Images/Elon musk.jpg')
#imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('Images/Bill gates.jpg')
#imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)