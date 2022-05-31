import cv2
from ui import analysis

vs = cv2.VideoCapture(0)
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    cv2.imshow('fruit',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('testpicture/fruit1.png',frame)
        break
# Stop video
vs.release()
# Close all started windows
cv2.destroyAllWindows()

analysis()
