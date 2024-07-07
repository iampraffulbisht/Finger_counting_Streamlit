import cv2 as cv
import numpy as np
import mediapipe
import hand_tracking_module as htm
import os
import time
import streamlit as st

# Initialize Streamlit app
st.set_page_config(page_title="Finger Counting App")
st.title("Finger Counting App")
st.subheader("Face your right hand plam side towards webcam")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")
# Set up webcam
wCam, hCam = 1280, 720
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load images for overlay
folderPath = "Fingers"
myList = os.listdir(folderPath)
myList.sort()
overlayList = [cv.resize(cv.imread(f'{folderPath}/{imgPath}'), (200, 200)) for imgPath in myList]

# Load specific image for special case
specificImage = cv.imread("f.png")
specificImage = cv.resize(specificImage, (150, 200))

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75, maxHands=1)
tipIds = [4, 8, 12, 16, 20]
fingers=0
pTime=0

# Main Streamlit app loop
while cap.isOpened() and not stop_button_pressed:
    # Read frame from webcam
    success, img = cap.read()

    # Process hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # Overlay appropriate image based on finger count
        if fingers == [0, 0, 1, 0, 0]:
            h, w, c = specificImage.shape
            img[0:h, 0:w] = specificImage
            cv.rectangle(img, (0, 225), (350, 445), (128, 128, 128), cv.FILLED)
            cv.putText(img,"Buzz",(35,300),cv.FONT_HERSHEY_COMPLEX_SMALL,5,(0,0,0),5)
            cv.putText(img,"off",(35,400),cv.FONT_HERSHEY_COMPLEX_SMALL,5,(0,0,0),5)
        else:
            h, w, c = overlayList[totalFingers - 1].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]
            cv.rectangle(img, (0, 225), (150, 425), (128, 128, 128), cv.FILLED)
            cv.putText(img, str(totalFingers), (35, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)


    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (1100, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Convert image to RGB for Streamlit display
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    frame_placeholder.image(img,channels="RGB")

    # Check for 'q' key press or stop button in Streamlit
    if cv.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
# Release resources

cap.release()
cv.destroyAllWindows()
