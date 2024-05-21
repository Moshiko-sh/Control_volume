import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
import numpy as np
#import serial
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize serial communication on COM port 3
#ser = serial.Serial('COM3', 9600)

cap = cv2.VideoCapture(0)  # Capture from camera
pTime = 0
detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volB = 400
volpre = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosion(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 50 - 300
        # Volume range -65 - 0
        vol = np.interp(length, [10, 140], [minVol, maxVol])
        volB = np.interp(length, [10, 140], [400, 150])
        volpre = np.interp(length, [10, 140], [0, 100])
        Bright = np.interp(length, [10, 140], [0, 255])
        volume.SetMasterVolumeLevel(vol, None)
        print(int(length), vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

        # Send the Bright value to the serial port
        #ser.write(f'{int(Bright)}'.encode())

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volB)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volpre)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 255, 0), 3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
#ser.close()  # Close the serial port
