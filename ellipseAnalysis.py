import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

def createPlots(filename):

    with open(filename, 'rb') as fp:
        ellipses = pickle.load(fp)

    # Unit conversion: pixels to micrometer
    px2um = 0.4003
    # Frame to time
    fps = 55
    # Length of video
    n = len(ellipses)

    # Plot the angle and the major and minor axes of the elipse
    elFig, (axAx, angAx) = plt.subplots(2, 1, sharex=True)
    axAx.plot(np.linspace(0, n/fps, n),
              [px2um*e[1][0] for e in ellipses])
    axAx.plot(np.linspace(0, n/fps, n),
              [px2um*e[1][1] for e in ellipses])
    angAx.plot(np.linspace(0, n/fps, n),
               [e[2] for e in ellipses])

    axAx.set_ylabel("Axis length (μm)")
    angAx.set_ylabel("Angle (°)")
    angAx.set_xlabel("Time (s)")
    plt.savefig("EllipseLen-"+filename+".pdf")

    # Get time-bounds to perform fft
    points = plt.ginput(2, timeout=-1)

    # Plot the position
    xs = [px2um*e[0][0] for e in ellipses]
    ys = [px2um*e[0][1] for e in ellipses]
    cmap = plt.get_cmap("Reds")
    posFig, posAx = plt.subplots()
    posAx.scatter(xs, ys, color=cmap(np.linspace(0, 1, len(ellipses))))
    posAx.set_ylabel("y-position (μm)")
    posAx.set_xlabel("x-position (μm)")
    plt.savefig("EllipsePos-"+filename+".pdf")

    # If not enough points were created, just show and return
    if len(points) < 2:
        plt.show()
        return

    # Otherwise, perform an fft of the x and y components of the position
    start_frame = int(points[0][0] * fps)
    end_frame = int(points[1][0] * fps)+1
    xs_levitating = xs[start_frame:end_frame]
    ys_levitating = ys[start_frame:end_frame]
    xfft = np.fft.rfft(xs_levitating)
    yfft = np.fft.rfft(ys_levitating)

    fftFig, fftAx = plt.subplots()
    fftAx.plot(xfft, label="FFT of x")
    fftAx.plot(yfft, label="FFT of y")
    fftAx.legend()

    plt.show()



if __name__ == '__main__':

    createPlots("data_20210316_Ringdown_850mA-0000.txt")
