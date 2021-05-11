import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

def circular_mean(angles):
    """
    Takes a list of angles in radians and returns the circular mean of them
    """

    # Convert the angles to cartesian points on the unit circle
    cartesian = np.column_stack((np.cos(angles), np.sin(angles)))
    #breakpoint()

    # Find the mean of the cartesian coordinates
    mean_cart = np.mean(cartesian, axis=0)

    # Find the angle of the mean point
    mean_angle = np.arctan2(mean_cart[1], mean_cart[0])

    # And return it
    return mean_angle

def make_histogram():
    """
    Creates a histogram of the angle of all available ellipse data files
    """

    # Create a list to save mean angles and mean poss
    mean_angles = []
    mean_poss = []

    # Find the data files (takes all files ending in .txt)
    filenames = glob.glob("*.txt")

    for filename in filenames:

        # Load the data
        with open(filename, 'rb') as file:
            ellipses = pickle.load(file)

        # Find the range where it levitates
        # Plot the angle and the major and minor axes of the elipse
        elFig, (axAx, angAx) = plt.subplots(2, 1, sharex=True)
        axAx.plot([e[1][0] for e in ellipses])
        axAx.plot([e[1][1] for e in ellipses])
        angAx.plot([e[2] for e in ellipses])

        axAx.set_ylabel("Axis length (μm)")
        angAx.set_ylabel("Angle (°)")
        angAx.set_xlabel("Time (s)")

        axAx.set_title("Pick one point when the particle starts to levitate\nand one when it stops")

        points = plt.ginput(2, timeout=-1)
        plt.close()

        if not points:
            continue

        start_frame = int(points[0][0])
        end_frame = int(points[1][0])+1

        # Angles when it levitates are the ones between start and end frame
        angles = [e[2] for e in ellipses[start_frame:end_frame]]

        # Convert the 0-180 span of angles to 0-2 pi
        # This is because it wraps around at 180 rather than 360
        # so the mean needs to reflect this
        #breakpoint()
        rads = np.array(angles) * np.pi / 90

        mean_rad = circular_mean(rads)

        mean_angle = 90 * mean_rad / np.pi

        # Add it to list of mean angles
        mean_angles.append(mean_angle)

        # Calculate the mean position
        positions = np.array([e[0] for e in ellipses[start_frame:end_frame]])

        mean_pos = np.mean(positions, axis=0)

        mean_poss.append(mean_pos)


    # End for filename in filenames

    # Plot histogram
    ax = plt.subplot(projection="polar")

    rads = np.array(mean_angles) * np.pi / 90
    bars = ax.hist(rads, bins=72, bottom=0)
    # Set labels
    ax.set_thetagrids(tuple(range(0,360,30)), labels=[f"{a}°" for a in range(0, 180, 15)])
    ax.set_rgrids([0, 1, 2, 3])
    ax.set_title("Average angle of semi-major axis of ellipse")


    fig, posAx = plt.subplots()

    
    for p in mean_poss:
        posAx.plot(*p, 'o')

    posAx.set_xlabel(r"$x$-position (μm)")
    posAx.set_ylabel(r"$y$-position (μm)")
    posAx.set_title("Average position for the films")
    plt.show()


if __name__=="__main__":
    make_histogram()
