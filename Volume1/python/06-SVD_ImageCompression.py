
>>> import numpy as np
>>> from scipy import linalg as la

# Generate a random matrix and get its compact SVD via SciPy.
>>> A = np.random.random((10,5))
>>> U,s,Vh = la.svd(A, full_matrices=False)
>>> print(U.shape, s.shape, Vh.shape)
(10, 5) (5,) (5, 5)

# Verify that U is orthonormal, U Sigma Vh = A, and the rank is correct.
>>> np.allclose(U.T @ U, np.identity(5))
<<True>>
>>> np.allclose(U @ np.diag(s) @ Vh, A)
<<True>>
>>> np.linalg.matrix_rank(A) == len(s)
<<True>>

>>> A = np.random.random((20, 20))
>>> A.size
400

>>> from matplotlib import pyplot as plt

# Send the RGB values to the interval (0,1).
>>> image_gray = plt.imread("hubble_gray.jpg") / 255.
>>> image_gray.shape            # Grayscale images are 2-d arrays.
(1158, 1041)

>>> image_color = plt.imread("hubble.jpg") / 255.
>>> image_color.shape           # Color images are 3-d arrays.
(1158, 1041, 3)

# The final axis has 3 layers for red, green, and blue values.
>>> red_layer = image_color[:,:,0]
>>> red_layer.shape
(1158, 1041)

# Display a gray image.
>>> plt.imshow(red_layer, cmap="gray")
>>> plt.axis("off")             # Turn off axis ticks and labels.
>>> plt.show()

# Display a color image.
>>> plt.imshow(image_color)     # cmap=None by default.
>>> plt.axis("off")
>>> plt.show()

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_images(images):
    """Animate a sequence of images. The input is a list where each
    entry is an array that will be one frame of the animation.
    """
    fig = plt.figure()
    plt.axis("off")
    im = plt.imshow(images[0], animated=True)

    def update(index):
        plt.title("Rank {} Approximation".format(index))
        im.set_array(images[index])
        return im,              # Note the comma!

    a = FuncAnimation(fig, update, frames=len(images), blit=True)
    plt.show()
