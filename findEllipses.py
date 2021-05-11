import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from get_background import get_background
from betterBlur import blur
from circleCenter import findCircle

def findEllipseInFrame(frame, returnImage=False):
    """
    findEllipses takes a frame with a binary image and returns the 
    largest of the ellipses that fit to the exteriour contours of the image

    optionally, if returnImage is True, it also returns an image of the ellipses
    drawn in Green and the contours drawn in Blue
    """

    # First, find the contours of the image.

    # RETR_EXTERNAL retrives only the outermost contours.
    # CHAIN_APPROX_SIMPLE compressess horizontal, vertical and diagonal
    # segments and leaves only their end points.
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the ellipses for each contour,
    # but ignore any contours that are too small, less than 100 points
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        if c.shape[0] <= 100:
            continue
        minEllipse[i] = cv2.fitEllipse(c)

    # Get the largest ellipse (by area of circumscribing rectangle)
    filteredEllipses = [e for e in minEllipse if e is not None]
    largestEllipse = max(filteredEllipses,
                         key=lambda e: e[1][0]*e[1][1])

    # Make position a numpy array instead of a tuple
    largestEllipse = (np.array(largestEllipse[0]), largestEllipse[1], largestEllipse[2])

    # If no image is to be returned, return here
    if not returnImage:
        return largestEllipse


    # Otherwise, convert the image from grayscale to color
    im = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # And draw each contour and ellipse in it
    for i, c in enumerate(contours):
        if c.shape[0] <= 100:
            continue

        # contour
        cv2.drawContours(im, contours, i, (255, 0, 0))

        # ellipse
        cv2.ellipse(im, minEllipse[i], (0, 255, 0), 2)

    # Return
    return largestEllipse, im


def findEllipses(vidPath, outPath):
    """
    This function takes a path to a video,
    and a path where a new video will be saved.

    It takes the video and tries to extract ellipses from within it
    and returns these ellipses that hopefully correspond to the particle positions.
    """

    # First, blur the video to remove some noise.
    blur(vidPath, "tmpBlurred.avi", show=False)

    # Second, extract the background of the video.
    # This requires manual intervention
    bg = get_background("tmpBlurred.avi")

    # Third, (also manually) get the center of the coordinate system
    # and the direction of the x-axis.
    # The positions of the ellipses will be relative to this.
    plt.imshow(bg)
    plt.title("Pick three points on the rim\n to calibrate the coordinate system.\nThe first two should make a line in the direction of the prefered x direction")
    points = plt.ginput(3, timeout=-1)
    plt.close()
    circleCenter = findCircle(points)
    xdir = (points[0][0] - points[1][0], points[0][1] - points[1][1])
    rotationAngle = np.arctan2(xdir[1], xdir[0])


    # Open the blurred video
    cap = cv2.VideoCapture("tmpBlurred.avi")

    # Get the first frame,
    # It is retrived here instead of in the while loop because the
    # writer needs to know the size of the frames it is going to write.
    ret, frame = cap.read()
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]

    # Create the writer
    # NOTE: If you are running on windows, you may have to change the encoding
    # MJPG to something else, maybe DIVX
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outPath, fourcc, 55.0, (fwidth, fheight), True)

    # Create a list of the ellipses that will be returned
    ellipses = []

    # Loop through the frames
    while ret:

        # First, convert the frame from colour to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Subtract the frame from the background
        # If the background is darker, set it to zero
        # The frame has to be converted to float as uint wraps around (3-5=254)
        gray = gray.astype(float)
        gray = bg - gray
        gray = np.where(gray <= 0, 0, gray)
        gray = gray.astype(np.uint8)

        # Blur it yet again
        gray = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)

        # ball is one tenth of the screen hight => ~1/100th of the screen area
        # Only the brightest couple of percent of pixels should be kept
        thresh = np.percentile(gray, 95)

        #ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        ret, gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # Now, find the largest ellipse in the frame
        ellipse, im = findEllipseInFrame(gray, returnImage=True)

        # Subtract the position of the center from the ellipse coordinates
        # And rotate so that the x-axis is aligned
        rotMat = np.array([[np.cos(-rotationAngle), -np.sin(-rotationAngle)],
                            [np.sin(-rotationAngle), np.cos(-rotationAngle)]])
        newPos = np.dot(rotMat, (ellipse[0] - circleCenter))
        # The angle is at first the angle clockwise from the y-axis
        # Convert to angle from new y-axis in the positive direction
        newAngle = -ellipse[2] - rotationAngle*180/np.pi
        ellipse = (newPos, ellipse[1], newAngle)

        # Save the ellipse to the list
        ellipses.append(ellipse)

        # Write the image to the file and show it on the screen
        out.write(im)
        cv2.imshow("Frame", im)

        k = cv2.waitKey(50) & 0xff
        if k == 27:     # Esc
            break

        # Get the next frame
        # if there is none, ret will be False, breaking the loop
        ret, frame = cap.read()

    # When while loop is ended,
    # Close the videos and windows and return the ellipses
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return ellipses

if __name__ == "__main__":

    # Name of the video
    # Should NOT contain ".avi"
    filename = "20210316_Ringdown_750mA_2021-03-16-190724-0000"

    # Find the ellipses
    ellipses = findEllipses(filename+".avi", "tmpFin.avi")

    # Save them to a file
    with open("data_" + filename + ".txt", 'wb') as fp:
        pickle.dump(ellipses, fp)
