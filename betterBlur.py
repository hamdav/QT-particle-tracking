import cv2
import numpy as np


def blur(vidPath, outPath, show=False):

    cap = cv2.VideoCapture(vidPath)
    ret, frame = cap.read()
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]
    print(fwidth , fheight)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outPath, fourcc, 55.0, (fwidth,fheight), 0)

    while ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #gray = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)
        #gray = cv2.medianBlur(gray, 11)
        gray = cv2.bilateralFilter(gray, 11, 75, 75)

        out.write(gray)

        if show:
            cv2.imshow("frame", gray)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        ret, frame = cap.read()


    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    blur("20210316_Ringdown_850mA-0000.avi", "tmpBlurred2.avi", show=True)
