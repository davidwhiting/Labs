
# Example of complex rounding error when using the fft and ifft. 
>>> x = np.array([1,2,3,4,5,6])
>>> y = sp.ifft(sp.fft(a)) 
#Result is close to x but complex components are present due to rounding error. 
>>> y                       
array([ 1. +0.00000000e+00j,  2. -1.11022302e-15j,  3. +1.69171391e-15j,
        4. +1.71752100e-17j,  5. -3.59446285e-16j,  6. -2.39219815e-16j])
# Take the real part to get the desired result. 
>>> np.real(y)              
array([ 1.,  2.,  3.,  4.,  5.,  6.])

# Create 2 seconds of mono white noise.
samplerate = 22050
noise = np.int16(np.random.randint(-32767, 32767, samplerate*2))

def naive_convolve(sample1,sample2):
    sig1 = np.append(sample1, np.zeros(len(sample2)-1))
    sig2 = np.append(sample2, np.zeros(len(sample1)-1))
    
    final = np.zeros_like(sig1)
    rsig1 = sig1[::-1]
    for k in range(len(sig1)-1):
        final[i+1] = np.sum((np.append(rsig1[i:],rsig1[::-i][:i]))*sig2)
    return final    

# Find the indices of the DFT that corresponds with the frequencies 1250 and 2500.
>>> low = 1250*len(samples)//rate
>>> high = 2500*len(samples)//rate

# Set the chosen coefficients between low and high to 0.
>>> fft_sig[low:high]=0
>>> fft_sig[-high:-low]=0

>>> A = np.array([[5, 3, 1], [4, 2, 7], [8, 9, 3]])
# Calculate the Fourier transform of A.
>>> fft = np.fft.fft2(A)
# Calculate the inverse Fourier transform.
>>> ifft = np.fft.ifft2(fft)
>>> np.allclose(A, ifft)
True

# Plot the blurry image (figure 1.3(a)).
>>> from scipy.misc import imread
>>> image = imread("face.png", True)
>>> plt.imshow(image, cmap='gray')
>>> plt.show()

# Plot the Fourier transform of the blurry image (figure 1.3(b)).
>>> fft = np.fft.fft2(image)
>>> plt.imshow(np.log(np.abs(fft)), cmap='gray')
>>> plt.show()

# Cover the spikes in the Fourier transform (figure 1.3(d)).
>>> fft[30:40, 97:107] = np.ones((10,10)) * fft[33][50]
>>> fft[-39:-29, -106:-96] = np.ones((10,10)) * fft[33][50]
>>> plt.imshow(np.log(np.abs(fft)),cmap='gray')
>>> plt.show()

# Plot the new image (figure 1.3(c)).
>>> new_image = np.abs(np.fft.ifft2(fft))
>>> plt.imshow(new_image, cmap='gray')
>>> plt.show()

>>> image = plt.imread('cameraman.jpg')
>>> plt.imshow(image, cmap = 'gray')
>>> plt.show()

1. def Filter(image, F):
2.     m, n = image.shape
3.     h, k = F.shape

 # Create a larger matrix of zeros
image_pad = np.zeros((m+2, n+2))
# Make the interior of image_pad equal to the original image
image_pad[1:1+m, 1:1+n] = image

5.    image_pad = # Create an array of zeros of the appropriate size
6.   # Make the interior of image_pad equal to image

7.    C = np.zeros(image.shape)
8.    for i in range(m):
9.        for j in range(n):
10.            C[i,j] = # Compute C[i, j]
