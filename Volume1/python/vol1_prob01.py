import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin

def stretch(data, a, b=None):
    """Linear Transformation: Stretch
    Compute the stretched version of an image, with matrix [[a, 0], [0, b]]
    If a=b, then it is a dilation
    """
    if b is None:
        b = a
    X = np.array([[a, 0], [0, b]])
    result = X.dot(data)
    return(result)

def shear(data, a, horizontal=True):
    """Linear Transformation: Shear
    Slants the vector by a scalar factor horizontally (x-axis) or vertically (y-axis). 
    """
    if horizontal:
        X = np.array([[1, a], [0, 1]])
    else:
        X = np.array([[1, 0], [a, 1]])
    result = X.dot(data)
    return(result)

def reflection(data, a, b):
    """Linear Transformation: Reflection
    Reflects the vector about a line that passes through the origin
    """
    X = np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])/(a**2 + b**2)
    result = X.dot(data)
    return(result)

def rotation(data, theta):
    """Linear Transformation: Rotation
    Rotates the vector about a line that passes through the origin
    """
    X = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    result = X.dot(data)
    return(result)

# Load the array from the .npy file.
data = np.load("data/horse.npy")

# Plot the x row against the y row with black pixels.
plt.plot(data[0], data[1], 'k,')

# Set the window limits to [-1, 1] by [-1, 1] and make the window square.
plt.axis([-1, 1, -1, 1])
plt.gca().set_aspect("equal")
plt.show()
