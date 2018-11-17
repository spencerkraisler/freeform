import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

i = 0
while(i < 100):
    # Capture frame-by-frame
    _, frame = cap.read()

    cv2.imwrite("./index/" + "IMG_" + str(i) + ".jpg", frame)
    i += 1

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()