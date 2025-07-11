import dlib
import sys
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import queue
import os
import platform
import subprocess

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

thresh = 0.27
# Try to find the model file in common locations
model_filename = "shape_predictor_68_face_landmarks.dat"
possible_paths = [
    os.path.join(os.path.dirname(__file__), model_filename),
    os.path.join(os.path.dirname(__file__), "models", model_filename),
    os.path.join(os.path.expanduser("~"), "Downloads", "model", model_filename),
    model_filename  # Current directory
]

modelPath = None
for path in possible_paths:
    if os.path.exists(path):
        modelPath = path
        break

if modelPath is None:
    print("ERROR: Could not find the dlib face landmark model file!")
    print("Please download 'shape_predictor_68_face_landmarks.dat' from:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract it and place it in one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

sound_path = os.path.join(os.path.dirname(__file__), "alarm.wav")
if not os.path.exists(sound_path):
    print("WARNING: alarm.wav not found. Sound alerts will be disabled.")
    sound_path = None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Eye landmark indices for 68-point model (standard dlib model)
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1200ms
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def gamma_correction(image):
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray) 

def soundAlert(path, threadStatusQ):
    if path is None:
        return
    
    try:
        # Convert to absolute path to avoid issues
        abs_path = os.path.abspath(path)
        
        # Try different methods based on platform
        if platform.system() == "Windows":
            # Use Windows winsound for better compatibility
            import winsound
            # Play the sound in a loop until stopped
            while True:
                if not threadStatusQ.empty():
                    FINISHED = threadStatusQ.get()
                    if FINISHED:
                        break
                try:
                    winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                    time.sleep(0.5)  # Short pause between repeats
                except Exception as e:
                    print(f"Error playing sound with winsound: {e}")
                    break
        else:
            # Use playsound for other platforms
            while True:
                if not threadStatusQ.empty():
                    FINISHED = threadStatusQ.get()
                    if FINISHED:
                        break
                try:
                    playsound.playsound(abs_path)
                    time.sleep(0.5)  # Short pause between repeats
                except Exception as e:
                    print(f"Error playing sound with playsound: {e}")
                    break
                    
    except Exception as e:
        print(f"Sound alert error: {e}")
        # Fallback: try system command once
        try:
            if platform.system() == "Windows":
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{abs_path}').PlaySync()"], 
                             capture_output=True, timeout=5)
            else:
                subprocess.run(["aplay", abs_path], capture_output=True, timeout=5)
        except Exception as e2:
            print(f"Fallback sound method also failed: {e2}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    
    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1  # 1 -> Open, 0 -> closed
    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus  

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if(state >= 0 and state <= falseBlinkLimit):
        if(eyeStatus):
            state = 0
        else:
            state += 1
    elif(state >= falseBlinkLimit and state < drowsyLimit):
        if(eyeStatus):
            blinkCount += 1
            state = 0
        else:
            state += 1
    else:
        if(eyeStatus):
            state = 0
            drowsy = 1
            blinkCount += 1
        else:
            drowsy = 1

def getLandmarks(im):
    imSmall = cv2.resize(im, None, 
                            fx=1.0 / FACE_DOWNSAMPLE_RATIO, 
                            fy=1.0 / FACE_DOWNSAMPLE_RATIO, 
                            interpolation=cv2.INTER_LINEAR)

    # Ensure the image is in BGR format (3 channels) before converting to grayscale
    if len(imSmall.shape) == 3:  # Check if it's a 3-channel (BGR) image
        imSmall = cv2.cvtColor(imSmall, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    elif len(imSmall.shape) == 2:  # If it's already grayscale, skip conversion
        pass
    
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("ERROR: Could not open camera!")
    print("Please check if:")
    print("1. Camera is connected properly")
    print("2. Camera is not being used by another application")
    print("3. You have camera permissions")
    sys.exit(1)

for i in range(10):
    ret, frame = capture.read()

totalTime = 0.0
validFrames = 0
dummyFrames = 100

print("Caliberation in Progress!")
while(validFrames < dummyFrames):
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    frame = cv2.resize(frame, None, 
                        fx=1 / IMAGE_RESIZE, 
                        fy=1 / IMAGE_RESIZE, 
                        interpolation=cv2.INTER_LINEAR)

    adjusted = histogram_equalization(frame)

    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Blink Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit()

    else:
        totalTime += timeLandmarks

print("Caliberation Complete!")

spf = totalTime / dummyFrames
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

drowsyLimit = drowsyTime / spf
falseBlinkLimit = blinkTime / spf
print("drowsy limit: {}, false blink limit: {}".format(drowsyLimit, falseBlinkLimit))

if __name__ == "__main__":
    # Initialize video writer after we have a frame
    vid_writer = None
    
    while(1):
        try:
            t = time.time()
            ret, frame = capture.read()
            if not ret:
                print("ERROR: Could not read frame from camera")
                break
                
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None, 
                                fx=1 / IMAGE_RESIZE, 
                                fy=1 / IMAGE_RESIZE, 
                                interpolation=cv2.INTER_LINEAR)

            # Initialize video writer with correct frame dimensions
            if vid_writer is None:
                vid_writer = cv2.VideoWriter('output-low-light-2.avi', 
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                           15, (frame.shape[1], frame.shape[0]))

            adjusted = histogram_equalization(frame)

            landmarks = getLandmarks(adjusted)
            if landmarks == 0:
                validFrames -= 1
                cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Blink Detection Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if drowsy:
                cv2.putText(frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not ALARM_ON and sound_path is not None:
                    print("DROWSINESS DETECTED - TRIGGERING ALARM!")
                    ALARM_ON = True
                    # Clear the queue before starting new alarm
                    while not threadStatusQ.empty():
                        threadStatusQ.get()
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.daemon = True
                    thread.start()
                elif sound_path is None:
                    print("DROWSINESS DETECTED - No alarm sound file!")
                    ALARM_ON = True  # Set to True to prevent repeated warnings

            else:
                cv2.putText(frame, "Blinks : {}".format(blinkCount), (460, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if ALARM_ON:
                    print("Stopping alarm - no longer drowsy")
                    # Stop the alarm when no longer drowsy
                    threadStatusQ.put(True)
                    ALARM_ON = False

            cv2.imshow("Blink Detection Demo", frame)
            if vid_writer is not None:
                vid_writer.write(frame)

            k = cv2.waitKey(1) 
            if k == ord('r'):
                state = 0
                drowsy = 0
                if ALARM_ON:
                    threadStatusQ.put(True)
                    ALARM_ON = False

            elif k == 27:
                break

        except Exception as e:
            print(e)

    capture.release()
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()
