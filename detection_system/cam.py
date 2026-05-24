import cv2


cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if not cap0.isOpened():
    print("Camera 0 failed")
if not cap1.isOpened():
    print("Camera 1 failed")

while True:
    r0, f0 = cap0.read()
    r1, f1 = cap1.read()

    if r0:
        cv2.imshow("CAM 0", f0)
    if r1:
        cv2.imshow("CAM 1", f1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()