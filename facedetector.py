import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open(r"C:\Users\edwar\Documents\Coding\Python\facialrecognitioncode\face_enc", "rb").read())

video_capture = cv2.VideoCapture(r"C:\Users\edwar\Pictures\Camera Roll\vid.mp4")
filepath = r'C:\Users\edwar\Documents\Coding\Python\Videos'
vidname = filepath + '\output_1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(vidname, fourcc, 10, (1920, 1080))

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    detection_result, rejectLevels, levelWeights = faceCascade.detectMultiScale3(frame, scaleFactor=1.0485258, minNeighbors=6,outputRejectLevels = 1)
    print(rejectLevels)
    print(levelWeights)
    print(detection_result)
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
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

out.release()
video_capture.release()
cv2.destroyAllWindows()
