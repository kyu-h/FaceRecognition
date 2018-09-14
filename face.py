from imutils import face_utils
from imutils.face_utils import FaceAligner
import time
import cv2
import dlib

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

print("[INFO] camera sensor warming up...")
vs = cv2.VideoCapture("ourface.mp4")
time.sleep(2.0)
prevTime = 0
fps = 0
i = 0
avgfps = 0

while True:
    ret, frame = vs.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    # fps = 1 / sec
    # avgfps += fps
    # cv2.putText(frame, "FPS : {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord('s'):
        if len(rects) > 0:
            faceAligned = fa.align(frame, gray_frame, rects[0])
            cv2.imshow("frame", faceAligned)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
