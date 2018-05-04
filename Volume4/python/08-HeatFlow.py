
import numpy as np
from matplotlib import animation, pyplot as plt

def sine_animation(res=100):
    # Make the x and y data.
    x = np.linspace(-1, 1, res+1)[:-1]
    y = np.sin(np.pi * x)
    # Initialize a matplotlib figure.
    f = plt.figure()
    # Set the x and y axes by constructing an axes object.
    plt.axes(xlim=(-1,1), ylim=(-1,1))
    # Plot an empty line to use in the animation.
    # Notice that we are unpacking a tuple of length 1.
    line, = plt.plot([], [])
    # Define an animation function that will update the line to
    # reflect the desired data for the i'th frame.
    def animate(i):
        # Set the data for updated version of the line.
        line.set_data(x, np.roll(y, i))
        # Notice that this returns a tuple of length 1.
        return line,
    # Create the animation object.
    # 'frames' is the number of frames before the animation should repeat.
    # 'interval' is the amount of time to wait before updating the plot.
    # Be sure to assign the animation a name so that Python does not
    # immediately garbage collect (delete) the object.
    a = animation.FuncAnimation(f, animate, frames=y.size, interval=20)
    # Show the animation.
    plt.show()

# Run the animation function we just defined.
sine_animation()

%def tridiag(a, b, c, x):
%    # Overrides c and x.
%    # The contents of x after computation will be the solution to the system.
%    size = x.size
%    temp = 0.
%    c[0] = c[0] / b[0]
%    x[0] = x[0] / b[0]
%    for n in range(size-2):
%        temp = 1. / (b[n+1] - a[n]*c[n])
%        c[n+1] *= temp
%        x[n+1] = (x[n+1] - a[n]*x[n]) * temp
%    x[size-1] = (x[size-1] - a[size-2]*x[size-2]) / (b[size-1] - a[size-2]*c[size-2])
%    for n in range(b.size-2, -1, -1):
%        x[n] = x[n] - c[n] * x[n+1]
%