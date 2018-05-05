
A = np.array([[0,1,0,0,1,0],[1,0,1,0,1,0],
                  [0,1,0,1,0,0],[0,0,1,0,1,1],
                  [1,1,0,1,0,0],[0,0,0,1,0,0]])

np.linalg.matrix_power(A,2)
np.linalg.matrix_power(A, 6)

def sparse_generator(n, c):
    """Return a symmetric nxn matrix with sparsity determined by c."""
    A = np.random.rand(n**2).reshape((n, n))
    A = ( A > c**(.5) )
    return A.T @ A

from scipy.misc import imread
from matplotlib import pyplot as plt

image = imread("dream.png")     # Read a (very) small image.
print(image.shape)              # Since the array is 3-dimensional, this is a color image.

# The image is read in as integers from 0 to 255.
print(image.<<min>>(), image.<<max>>(), image.dtype)

# Scale the image to floats between 0 and 1 for Matplotlib.
scaled = image / 255.
print(scaled.<<min>>(), scaled.<<max>>(), scaled.dtype)

# Display the scaled image.
plt.imshow(scaled)
plt.axis("off")
plt.show()

# Average the RGB values of a colored image to obtain a grayscale image.
brightness = scaled.mean(axis=2)        # Average over the last axis.
print(brightness.shape)                 # Note that the array is now 2-D.

# Display the image in gray.
plt.imshow(brightness, cmap="gray")
plt.axis("off")
plt.show()

import numpy as np
A = np.random.randint(0, 10, (3,4))
print(A)

# Unravel the 2-D array (by rows) into a 1-D array.
np.ravel(A)

# Unravel a grayscale image into a 1-D array and check its size.
M,N = brightness.shape
flat_brightness = np.ravel(brightness)
M*N == flat_brightness.size
print(flat_brightness.shape)

def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.<<int>>), R[mask]

# Compute the neighbors and corresponding distances from the figure.
neighbors_1, distances_1 = get_neighbors(5, 1.2, 4, 4)
print(neighbors_1, distances_1, sep='\n')

# Increasing the radius from 1.2 to 1.5 results in more neighbors.
neighbors_2, distances_2 = get_neighbors(5, 1.5, 4, 4)
print(neighbors_2, distances_2, sep='\n')

x = np.arange(-5,5).reshape((5,2)).T
print(x)

# Construct a boolean mask of x describing which entries of x are positive.
mask = x > 0
print(mask)

# Use the mask to zero out all of the nonpositive entries of x.
x * mask

mask = np.arange(-5,5).reshape((5,2)).T > 0
print(mask)

# The mask can be negated with the tilde operator ~.
print(~mask)

# Stack a mask into a 3-D array with np.dstack().
print(mask.shape, np.dstack((mask, mask, mask)).shape)
