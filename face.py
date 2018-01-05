import cv2
import numpy as np
import time

detect_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
overlay = cv2.imread("logo.png", -1)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = detect_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in detected:
        resized = cv2.resize(overlay, (w, h), interpolation = cv2.INTER_CUBIC)

        for c in range(0,3):
            alpha = resized[:, :, 3] / 255.0
            color = resized[:, :, c] * (alpha)
            beta  = img[y:y+h, x:x+w, c] * (1.0-alpha)

            img[y:y+h, x:x+w, c] = color + beta

    cv2.imshow('img', img)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"): #Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
