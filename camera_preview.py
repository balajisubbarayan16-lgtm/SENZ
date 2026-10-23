import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open the webcam. Check that it is connected and not in use by another app.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    cv2.imshow("SPONGE — press Q to close", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
