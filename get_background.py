import cv2

import numpy as np

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def get_background(vidPath):
    """
    Takes a video path and opens a selection
    making it possible to extract a background from the 
    video by marking which areas to ignore. 
    """

    print("""
        Select frame with:
        n       - next frame
        p       - previous frame
        e       - goto last frame
        0       - goto first frame
        ENTER   - type frame number in terminal
        ESC     - Done

        On a frame, you can click once with mouse to set center, 
        and again to set radius. If you are not happy with the result
        click again to set new center and radius. The old circle(s)
        will stay visible until you switch frames, 
        but will then be discarded.

        Make sure you have at least one frame where the background
        is not obscured for each bit of background before pressing ESC.
        I usually aim for 3+ frames with the ball in one position
        (usually the first three frames) and 3+ frames with the ball
        in another position (usually the last three frames)
    """)

    frame_no = 0

    circles = dict()
    deleted_circles = []

    #define the event
    def getxy(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN :
            if frame_no not in circles or circles[frame_no][0] != 0:
                circles[frame_no] = [0, (x, y)]
            else:
                _, (cx, cy) = circles[frame_no]
                circles[frame_no][0] = int(np.sqrt( (cx-x)**2 + (cy-y)**2 ))


    #Read the videos first and last frames
    cap = cv2.VideoCapture(vidPath)

    # Get first frame
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get total number of frames
    no_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    #Set mouse CallBack event
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', getxy)
 
    #show the image
    while True:
        cv2.imshow('image',frame)
        if frame_no in circles:
            cv2.circle(frame, circles[frame_no][1], circles[frame_no][0], (0, 255, 0), 2)

        k = cv2.waitKey(50) & 0xFF
        #print(k)
        to_frame = -1

        if k == 27:     # Esc
            break
        elif k == 13:     # Enter: Go to last frame (if not already there)
            try:
                to_frame = int(input("Go to frame: "))
            except ValueError:
                to_frame = -1

            if not (0 <= to_frame < no_of_frames):
                print("Invalid frame, must be between 0 and", no_of_frames)

        elif k == ord("n"):
            if frame_no + 1 < no_of_frames:
                to_frame = frame_no + 1
            else:
                print("No next frame")
        elif k == ord("p"):
            if frame_no - 1 >= 0:
                to_frame = frame_no - 1
            else:
                print("No previous frame")
        elif k == ord("0"):
            to_frame = 0
        elif k == ord("e"):
            to_frame = no_of_frames - 1

        # Go to frame if possible
        if 0 <= to_frame < no_of_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, to_frame)
            frame_no = to_frame
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the background

    xs = np.arange(frame.shape[1])
    ys = np.arange(frame.shape[0])
    X, Y = np.meshgrid(xs, ys)
    bg = np.zeros(frame.shape)
    counts = np.zeros(frame.shape)
    for frame_no in circles:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r, (cx, cy) = circles[frame_no]

        # The mask is 1 everywhere outside the circle and 0 inside it
        mask = (cx-X)**2 + (cy-Y)**2 >= r**2

        bg = np.where(mask, (counts * bg + frame) / (counts + 1), bg)
        counts += mask

    cv2.imshow("image", bg.astype(np.uint8))

    print(circles)
    return bg


if __name__=="__main__":
    get_background('20210316_Ringdown_850mA-0000.avi')

