import tkinter as tk
import face_recognition
import imutils
import pickle
import time
import cv2
import os

faceCascadePath = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(faceCascadePath)
data = pickle.loads(open(r"C:\Users\edwar\Documents\Coding\Python\facialrecognitioncode\face_enc", "rb").read())

camera = cv2.VideoCapture(0)

my_w = tk.Tk()
my_w.geometry("1530x780")

c_v1 = tk.IntVar()
c1 = tk.Checkbutton(my_w,text='Edward Simpson',variable=c_v1,
	onvalue=1,offvalue=0)
c1.grid(column = 0, row = 0)

c_v2 = tk.IntVar()
c2 = tk.Checkbutton(my_w,text='Brian Simpson',variable=c_v2,
	onvalue=1,offvalue=0)
c2.grid(column = 1, row = 0)

delayms = 100

def setButtons(string):
    if(string == "Edward Simpson"):
        c_v1.set(True)
    if(string == "Brian Simpson"):
        c_v2.set(True)

def checkFaces():
    ret, frame = camera.read()
    frame = cv2.resize(frame, (300, 200))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        setButtons(name)
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
        print(name)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        camera.release()
        cv2.destroyAllWindows()
    my_w.after(delayms,checkFaces)
        
c_v1.set(0)
my_w.after(delayms,checkFaces)
my_w.mainloop()
