
>>> from scipy.signal import fftconvolve
>>> # Initialize a filter.
>>> L = np.ones(2)/np.sqrt(2)
>>> # Initialize a signal X.
>>> X = np.sin(np.linspace(0,2*np.pi,16))
>>> # Convolve X with L.
>>> fftconvolve(X, L)
[ -1.84945741e-16   2.87606238e-01   8.13088984e-01   1.19798126e+00
   1.37573169e+00   1.31560561e+00   1.02799937e+00   5.62642704e-01
   7.87132986e-16  -5.62642704e-01  -1.02799937e+00  -1.31560561e+00
  -1.37573169e+00  -1.19798126e+00  -8.13088984e-01  -2.87606238e-01
  -1.84945741e-16]

>>> # Downsample an array X.
>>> sampled = X[1::2]

domain = np.linspace(0, 4*np.pi, 1024)
noise =  np.random.randn(1024)*.1
noisysin = np.sin(domain) + noise
coeffs = dwt(noisysin, L, H, 4)

>>> # Upsample the coefficient arrays A and D.
>>> up_A = np.zeros(2*A.size)
>>> up_A[::2] = A
>>> up_D = np.zeros(2*D.size)
>>> up_D[::2] = D
>>> # Convolve and add, discarding the last entry.
>>> A = fftconvolve(up_A, L)[:-1] + fftconvolve(up_D, H)[:-1]

$ pip install PyWavelets

$ conda install -c ioos pywavelets=0.4.0

>>> from scipy.misc import imread
>>> import pywt                             # The PyWavelets package.
# The True parameter produces a grayscale image.
>>> mandrill = imread('mandrill1.png', True)
# Use the Daubechies 4 wavelet with periodic extension.
>>> lw = pywt.dwt2(mandrill, 'db4', mode='per')

>>> plt.subplot(221)
>>> plt.imshow(lw[0], cmap='gray')
>>> plt.axis('off')
>>> plt.subplot(222)
# The absolute value of the detail subbands is plotted to highlight contrast.
>>> plt.imshow(np.abs(lw[1][0]), cmap='gray')
>>> plt.axis('off')
>>> plt.subplot(223)
>>> plt.imshow(np.abs(lw[1][1]), cmap='gray')
>>> plt.axis('off')
>>> plt.subplot(224)
>>> plt.imshow(np.abs(lw[1][2]), cmap='gray')
>>> plt.axis('off')
>>> plt.subplots_adjust(wspace=0, hspace=0)      # Remove space between plots.

>>> # List the available wavelet families.
>>> print(pywt.families())
['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
>>> # List the available wavelets in a given family.
>>> print(pywt.wavelist('coif'))
['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']

image = imread(filename,True)
wavelet = pywt.Wavelet('haar')
WaveletCoeffs = pywt.wavedec2(image,wavelet)
new_image = pywt.waverec2(WaveletCoeffs[:-1], wavelet)

>>> A = np.arange(-4,5).reshape(3,3)
>>> A
array([[-4, -3, -2],
       [-1,  0,  1],
       [ 2,  3,  4]])
>>> pywt.thresholding.hard(A,1.5)
array([[-4, -3, -2],
       [ 0,  0,  0],
       [ 2,  3,  4]])
>>> pywt.thresholding.soft(A,1.5)
array([[-2.5, -1.5, -0.5],
       [ 0. ,  0. ,  0. ],
       [ 0.5,  1.5,  2.5]])

>>> # assume we have an array M, numerical values a and b
>>> M.mean()
>>> M.max()
>>> M.min()
>>> max(a,b)

%>>> # assume X, Y are numpy arrays of same shape
%>>> m = X < -2 # create mask for entries less than -2
%>>> Y[m] = np.ceil(X[m]) + 2 # set corresponding entries of Y
%
$ pip install bitstring

>>> import bitstring as bs

>>> # assume subbands is my list of the 64 quantized subbands
>>> g1 = []     # this will hold the group 1 coefficients
>>> s1 = []     # keep track of the subband dimensions in group 1
>>> t1 = []     # keep track of which subbands were included
>>> for i in xrange(19):
>>>     s1.append(subbands[i].shape)
>>>     if subbands[i].any(): # True if any nonzero entry
>>>         g1.extend(subbands[i].ravel())
>>>         t1.append(True)
>>>     else: # the subband was not transmitted
>>>         t1.append(False)

>>> # reconstruct the subbands in group 1
>>> subbands1 = []     # the reconstructed subbands in group 1
>>> i = 0
>>> for j, shape in enumerate(s1):
>>>     if t1[j]: # if the j-th subband was included
>>>         l = shape[0]*shape[1] # number of entries in the subband
>>>         subbands1.append(np.array(g1[i:i+l]).reshape(shape))
>>>         i += l
>>>     else: # the j-th subband wasn't included, so all zeros
>>>         subbands1.append(np.zeros(shape))

>>> groups, self._shapes, self._tvals = self._group(q_subbands)

>>> import bitstring as bs

>>> bits = bs.BitArray()
>>> # add bit patterns 1101 and 01
>>> bits.append('0b1101')
>>> bits.append('0b01')

>>> # add the 8-bit integer 212, and then the 16-bit integer 1047
>>> bits.append('uint:8=212')
>>> bits.append('uint:16=1047')

>>> # view the entire bit string
>>> print bits.bin
110101110101000000010000010111

>>> bitreader = bs.ConstBitStream(bits)
>>> for i in xrange(6):
>>>     print bitreader.read('bin:1')
1
1
0
1
0
1

>>> print bitreader.read('uint:8')
212
>>> print bitreader.read('uint:16')
1047

>>> # assume groups is a list of the three groups of coefficients
>>> # for each group, get huffman indices, create huffman tree, and encode
>>> huff_maps = []
>>> bitstrings = []
>>> for i in xrange(3):
>>>     inds, freqs, extra = self._huffmanIndices(groups[i])
>>>     huff_map = huffman(freqs)
>>>     huff_maps.append(huff_map)
>>>     bitstrings.append(self._encode(inds, extra, huff_map))
>>>
>>> # store the bitstrings and the huffman maps
>>> self._bitstrings = bitstrings
>>> self._huff_maps = huff_maps

# Try out different values of r between .1 to .9.
r = .5
finger = imread('uncompressed_finger.png', True)
wsq = WSQ()
wsq.compress(finger, r)
print(wsq.get_ratio())
new_finger = wsq.decompress()
plt.subplot(211)
plt.imshow(finger, cmap=plt.cm.Greys_r)
plt.subplot(212)
plt.imshow(np.abs(new_finger), cmap=plt.cm.Greys_r)
plt.show()

class WSQ:
    """Perform image compression using the Wavelet Scalar Quantization
    algorithm. This class is a structure for performing the algorithm, to
    actually perform the compression and decompression, use the _compress
    and _decompress methods respectively. Note that all class attributes
    are set to None in __init__, but their values are initialized in the
    compress method.

    Attributes:
        _pixels (int): Number of pixels in source image.
        _s (float): Scale parameter for image preprocessing.
        _m (float): Shift parameter for image preprocessing.
        _Q ((16, ), ndarray): Quantization parameters q for each subband.
        _Z ((16, ), ndarray): Quantization parameters z for each subband.
        _bitstrings (list): List of 3 BitArrays, giving bit encodings for
            each group.
        _tvals (tuple): Tuple of 3 lists of bools, indicating which
            subbands in each groups were encoded.
        _shapes (tuple): Tuple of 3 lists of tuples, giving shapes of each
            subband in each group.
        _huff_maps (list): List of 3 dictionaries, mapping huffman index to
            bit pattern.
    """

    def __init__(self):
        self._pixels = None
        self._s = None
        self._m = None
        self._Q = None
        self._Z = None
        self._bitstrings = None
        self._tvals = None
        self._shapes= None
        self._huff_maps = None
        self._infoloss = None

    def compress(self, img, r, gamma=2.5):
        """The main compression routine. It computes and stores a bitstring
        representation of a compressed image, along with other values
        needed for decompression.

        Parameters:
            img ((m,n), ndarray): Numpy array containing 8-bit integer
                pixel values.
            r (float): Defines compression ratio. Between 0 and 1, smaller
                numbers mean greater levels of compression.
            gamma (float): A parameter used in quantization.
        """
        self._pixels = img.size   # Store image size.
        # Process then decompose image into subbands.
        mprime = self.pre_process(img)
        subbands = self.decompose(img)
        # Calculate quantization parameters, quantize the image then group.
        self._Q, self._Z = self.get_bins(subbands, r, gamma)
        q_subbands = [self.quantize(subbands[i],self._Q[i],self._Z[i])
                      for i in range(16)]
        groups, self._shapes, self._tvals = self.group(q_subbands)

        # Complete the Huffman encoding and transfer to bitstring.
        huff_maps = []
        bitstrings = []
        for i in range(3):
            inds, freqs, extra = self.huffman_indices(groups[i])
            huff_map = huffman(freqs)
            huff_maps.append(huff_map)
            bitstrings.append(self.encode(inds, extra, huff_map))

        # Store the bitstrings and the huffman maps.
        self._bitstrings = bitstrings
        self._huff_maps = huff_maps

    def pre_process(self, img):
        """Preprocessing routine that takes an image and shifts it so that
        roughly half of the values are on either side of zero and fall
        between -128 and 128.

        Parameters:
            img ((m,n), ndarray): Numpy array containing 8-bit integer
                pixel values.

        Returns:
            ((m,n), ndarray): Processed numpy array containing 8-bit
                integer pixel values.
        """
        pass

    def post_process(self, img):
        """Postprocess routine that reverses pre_process().

        Parameters:
            img ((m,n), ndarray): Numpy array containing 8-bit integer
                pixel values.

        Returns:
            ((m,n), ndarray): Unprocessed numpy array containing 8-bit
                integer pixel values.
        """
        pass

    def decompose(self, img):
        """Decompose an image into the WSQ subband pattern using the
        Coiflet1 wavelet.

        Parameters:
            img ((m,n) ndarray): Numpy array holding the image to be
                decomposed.

        Returns:
            subbands (list): List of 16 numpy arrays containing the WSQ
                subbands in order.
        """
        pass

    def recreate(self, subbands):
        """Recreate an image from the 16 WSQ subbands.

        Parameters:
            subbands (list): List of 16 numpy arrays containing the WSQ
                subbands in order.

        Returns:
            img ((m,n) ndarray): Numpy array, the image recreated from the
                WSQ subbands.
        """
        pass

    def get_bins(self, subbands, r, gamma):
        """Calculate quantization bin widths for each subband. These will
        be used to quantize the wavelet coefficients.

        Parameters:
            subbands (list): List of 16 WSQ subbands.
            r (float): Compression parameter, determines the degree of
                compression.
            gamma(float): Parameter used in compression algorithm.

        Returns:
            Q ((16, ) ndarray): Array of quantization step sizes.
            Z ((16, ) ndarray): Array of quantization coefficients.
        """
        subband_vars = np.zeros(16)
        fracs = np.zeros(16)

        for i in range(len(subbands)): # Compute subband variances.
            X,Y = subbands[i].shape
            fracs[i]=(X*Y)/(np.float(finger.shape[0]*finger.shape[1]))
            x = np.floor(X/8.).astype(int)
            y = np.floor(9*Y/32.).astype(int)
            Xp = np.floor(3*X/4.).astype(int)
            Yp = np.floor(7*Y/16.).astype(int)
            mu = subbands[i].mean()
            sigsq = (Xp*Yp-1.)**(-1)*((subbands[i][x:x+Xp, y:y+Yp]-mu)**2).sum()
            subband_vars[i] = sigsq

        A = np.ones(16)
        A[13], A[14] = [1.32]*2

        Qprime = np.zeros(16)
        mask = subband_vars >= 1.01
        Qprime[mask] = 10./(A[mask]*np.log(subband_vars[mask]))
        Qprime[:4] = 1
        Qprime[15] = 0

        K = []
        for i in range(15):
            if subband_vars[i] >= 1.01:
                K.append(i)

        while True:
            S = fracs[K].sum()
            P = ((np.sqrt(subband_vars[K])/Qprime[K])**fracs[K]).prod()
            q = (gamma**(-1))*(2**(r/S-1))*(P**(-1./S))
            E = []
            for i in K:
                if Qprime[i]/q >= 2*gamma*np.sqrt(subband_vars[i]):
                    E.append(i)
            if len(E) > 0:
                for i in E:
                    K.remove(i)
                continue
            break

        Q = np.zeros(16) # Final bin widths.
        for i in K:
            Q[i] = Qprime[i]/q
        Z = 1.2*Q

        return Q, Z

    def quantize(self, coeffs, Q, Z):
        """Implementation of a uniform quantizer which maps wavelet
        coefficients to integer values using the quantization parameters
        Q and Z.

        Parameters:
            coeffs ((m,n) ndarray): Contains the floating-point values to
                be quantized.
            Q (float): The step size of the quantization.
            Z (float): The null-zone width (of the center quantization bin).

        Returns
            out ((m,n) ndarray): Numpy array of the quantized values.
        """
        pass

    def dequantize(self, coeffs, Q, Z, C=0.44):
        """Given quantization parameters, approximately reverses the
        quantization effect carried out in quantize().

        Parameters:
            coeffs ((m,n) ndarray): Array of quantized coefficients.
            Q (float): The step size of the quantization.
            Z (float): The null-zone width (of the center quantization bin).
            C (float): Centering parameter, defaults to .44.

        Returns:
            out ((m,n) ndarray): Array of dequantized coefficients.
        """
        pass

    def group(self, subbands):
        """Split the quantized subbands into 3 groups.

        Parameters:
            subbands (list): Contains 16 numpy arrays which hold the
                quantized coefficients.

        Returns:
            gs (tuple): (g1,g2,g3) Each gi is a list of quantized coeffs
                for group i.
            ss (tuple): (s1,s2,s3) Each si is a list of tuples which
                contain the shapes for group i.
            ts (tuple): (s1,s2,s3) Each ti is a list of bools indicating
                which subbands were included.
        """
        g1 = [] # This will hold the group 1 coefficients.
        s1 = [] # Keep track of the subband dimensions in group 1.
        t1 = [] # Keep track of which subbands were included.
        for i in range(10):
            s1.append(subbands[i].shape)
            if subbands[i].any(): # True if there is any nonzero entry.
                g1.extend(subbands[i].ravel())
                t1.append(True)
            else: # The subband was not transmitted.
                t1.append(False)

        g2 = [] # This will hold the group 2 coefficients.
        s2 = [] # Keep track of the subband dimensions in group 2.
        t2 = [] # Keep track of which subbands were included.
        for i in range(10, 13):
            s2.append(subbands[i].shape)
            if subbands[i].any(): # True if there is any nonzero entry.
                g2.extend(subbands[i].ravel())
                t2.append(True)
            else: # The subband was not transmitted.
                t2.append(False)

        g3 = [] # This will hold the group 3 coefficients.
        s3 = [] # Keep track of the subband dimensions in group 3.
        t3 = [] # Keep track of which subbands were included.
        for i in range(13,16):
            s3.append(subbands[i].shape)
            if subbands[i].any(): # True if there is any nonzero entry.
                g3.extend(subbands[i].ravel())
                t3.append(True)
            else: # The subband was not transmitted.
                t3.append(False)

        return (g1,g2,g3), (s1,s2,s3), (t1,t2,t3)

    def ungroup(self, gs, ss, ts):
        """Re-create the subband list structure from the information stored
        in gs, ss and ts.

        Parameters:
            gs (tuple): (g1,g2,g3) Each gi is a list of quantized coeffs
                for group i.
            ss (tuple): (s1,s2,s3) Each si is a list of tuples which
                contain the shapes for group i.
            ts (tuple): (s1,s2,s3) Each ti is a list of bools indicating
                which subbands were included.

        Returns:
            subbands (list): Contains 16 numpy arrays holding quantized
                coefficients.
        """
        subbands1 = [] # The reconstructed subbands in group 1.
        i = 0
        for j, shape in enumerate(ss[0]):
            if ts[0][j]: # True if the j-th subband was included.
                l = shape[0]*shape[1] # Number of entries in the subband.
                subbands1.append(np.array(gs[0][i:i+l]).reshape(shape))
                i += l
            else: # The j-th subband wasn't included, so all zeros.
                subbands1.append(np.zeros(shape))

        subbands2 = [] # The reconstructed subbands in group 2.
        i = 0
        for j, shape in enumerate(ss[1]):
            if ts[1][j]: # True if the j-th subband was included.
                l = shape[0]*shape[1] # Number of entries in the subband.
                subbands2.append(np.array(gs[1][i:i+l]).reshape(shape))
                i += l
            else: # The j-th subband wasn't included, so all zeros.
                subbands2.append(np.zeros(shape))

        subbands3 = [] # the reconstructed subbands in group 3
        i = 0
        for j, shape in enumerate(ss[2]):
            if ts[2][j]: # True if the j-th subband was included.
                l = shape[0]*shape[1] # Number of entries in the subband.
                subbands3.append(np.array(gs[2][i:i+l]).reshape(shape))
                i += l
            else: # The j-th subband wasn't included, so all zeros.
                subbands3.append(np.zeros(shape))

        subbands1.extend(subbands2)
        subbands1.extend(subbands3)
        return subbands1

    def huffman_indices(self, coeffs):
        """Calculate the Huffman indices from the quantized coefficients.

        Parameters:
            coeffs (list): Integer values that represent quantized
                coefficients.

        Returns:
            inds (list): The Huffman indices.
            freqs (ndarray): Array whose i-th entry gives the frequency of
                index i.
            extra (list): Contains zero run lengths and coefficient
                magnitudes for exceptional cases.
        """
        N = len(coeffs)
        i = 0
        inds = []
        extra = []
        freqs = np.zeros(254)

        # Sweep through the quantized coefficients.
        while i < N:

            # First handle zero runs.
            zero_count = 0
            while coeffs[i] == 0:
                zero_count += 1
                i += 1
                if i >= N:
                    break

            if zero_count > 0 and zero_count < 101:
                inds.append(zero_count - 1)
                freqs[zero_count - 1] += 1
            elif zero_count >= 101 and zero_count < 256: # 8 bit zero run.
                inds.append(104)
                freqs[104] += 1
                extra.append(zero_count)
            elif zero_count >= 256: # 16 bit zero run.
                inds.append(105)
                freqs[105] += 1
                extra.append(zero_count)
            if i >= N:
                break

            # now handle nonzero coefficients
            if coeffs[i] > 74 and coeffs[i] < 256: # 8 bit pos coeff.
                inds.append(100)
                freqs[100] += 1
                extra.append(coeffs[i])
            elif coeffs[i] >= 256: # 16 bit pos coeff.
                inds.append(102)
                freqs[102] += 1
                extra.append(coeffs[i])
            elif coeffs[i] < -73 and coeffs[i] > -256: # 8 bit neg coeff.
                inds.append(101)
                freqs[101] += 1
                extra.append(abs(coeffs[i]))
            elif coeffs[i] <= -256: # 16 bit neg coeff.
                inds.append(103)
                freqs[103] += 1
                extra.append(abs(coeffs[i]))
            else: # Current value is a nonzero coefficient in the range [-73, 74].
                inds.append(179 + coeffs[i])
                freqs[179 + coeffs[i].astype(int)] += 1
            i += 1

        return list(map(int,inds)), list(map(int,freqs)), list(map(int,extra))

    def indices_to_coeffs(self, indices, extra):
        """Calculate the coefficients from the Huffman indices plus extra
        values.

        Parameters:
            indices (list): List of Huffman indices.
            extra (list): Indices corresponding to exceptional values.

        Returns:
            coeffs (list): Quantized coefficients recovered from the indices.
        """
        coeffs = []
        j = 0 # Index for extra array.

        for s in indices:
            if s < 100: # Zero count of 100 or less.
                coeffs.extend(np.zeros(s+1))
            elif s == 104 or s == 105: # Zero count of 8 or 16 bits.
                coeffs.extend(np.zeros(extra[j]))
                j += 1
            elif s in [100, 102]: # 8 or 16 bit pos coefficient.
                coeffs.append(extra[j]) # Get the coefficient from the extra list.
                j += 1
            elif s in [101, 103]: # 8 or 16 bit neg coefficient.
                coeffs.append(-extra[j]) # Get the coefficient from the extra list.
                j += 1
            else: # Coefficient from -73 to +74.
                coeffs.append(s-179)
        return coeffs

    def encode(self, indices, extra, huff_map):
        """Encodes the indices using the Huffman map, then returns
        the resulting bitstring.

        Parameters:
            indices (list): Huffman Indices.
            extra (list): Indices corresponding to exceptional values.
            huff_map (dict): Dictionary that maps Huffman index to bit
                pattern.

        Returns:
            bits (BitArray object): Contains bit representation of the
                Huffman indices.
        """
        bits = bs.BitArray()
        j = 0 # Index for extra array.
        for s in indices: # Encode each huffman index.
            bits.append('0b' + huff_map[s])

            # Encode extra values for exceptional cases.
            if s in [104, 100, 101]: # Encode as 8-bit ints.
                bits.append('uint:8={}'.format(int(extra[j])))
                j += 1
            elif s in [102, 103, 105]: # Encode as 16-bit ints.
                bits.append('uint:16={}'.format(int(extra[j])))
                j += 1
        return bits

    def decode(self, bits, huff_map):
        """Decodes the bits using the given huffman map, and returns
        the resulting indices.

        Parameters:
            bits (BitArray object): Contains bit-encoded Huffman indices.
            huff_map (dict): Maps huffman indices to bit pattern.

        Returns:
            indices (list): Decoded huffman indices.
            extra (list): Decoded values corresponding to exceptional indices.
        """
        indices = []
        extra = []

        # Reverse the huffman map to get the decoding map.
        dec_map = {v:k for k, v in huff_map.items()}

        # Wrap the bits in an object better suited to reading.
        bits = bs.ConstBitStream(bits)

        # Read each bit at a time, decoding as we go.
        i = 0 # The index of the current bit.
        pattern = '' # The current bit pattern.
        while i < bits.length:
            pattern += bits.read('bin:1') # Read in another bit.
            i += 1

            # Check if current pattern is in the decoding map.
            if pattern in dec_map:
                indices.append(dec_map[pattern]) # Insert huffman index.

                # If an exceptional index, read next bits for extra value.
                if dec_map[pattern] in (100, 101, 104): # 8-bit int or 8-bit zero run length.
                    extra.append(bits.read('uint:8'))
                    i += 8
                elif dec_map[pattern] in (102, 103, 105): # 16-bit int or 16-bit zero run length.
                    extra.append(bits.read('uint:16'))
                    i += 16
                pattern = '' # Reset the bit pattern.
        return indices, extra

    def decompress(self):
        """Return the uncompressed image recovered from the compressed
            bistring representation.

        Returns:
            img ((m,n) ndaray): The recovered, uncompressed image.
        """
        # For each group, decode the bits, map from indices to coefficients.
        groups = []
        for i in range(3):
            indices, extras = self.decode(self._bitstrings[i],
                                           self._huff_maps[i])
            groups.append(self.indices_to_coeffs(indices, extras))

        # Recover the subbands from the groups of coefficients.
        q_subbands = self.ungroup(groups, self._shapes, self._tvals)

        # Dequantize the subbands.
        subbands = [self.dequantize(q_subbands[i], self._Q[i], self._Z[i])
                    for i in range(16)]

        # Recreate the image.
        img = self.recreate(subbands)

        # Post-process, return the image.
        return self.post_process(img)

    def get_ratio(self):
        """Calculate the compression ratio achieved.

        Returns:
            ratio (float): Ratio of number of bytes in the original image
                to the number of bytes contained in the bitstrings.
        """
        pass

# Helper functions and classes for the Huffman encoding portions of WSQ algorithm.

import queue
class huffmanLeaf():
    """Leaf node for Huffman tree."""
    def __init__(self, symbol):
        self.symbol = symbol

    def makeMap(self, huff_map, path):
        huff_map[self.symbol] = path

    def __str__(self):
        return str(self.symbol)

    def __lt__(self,other):
        return False

class huffmanNode():
    """Internal node for Huffman tree."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def makeMap(self, huff_map, path):
        """Traverse the huffman tree to build the encoding map."""
        self.left.makeMap(huff_map, path + '0')
        self.right.makeMap(huff_map, path + '1')

    def __lt__(self,other):
        return False

def huffman(freqs):
    """
    Generate the huffman tree for the given symbol frequencies.
    Return the map from symbol to bit pattern.
    """
    q = queue.PriorityQueue()
    for i in range(len(freqs)):
        leaf = huffmanLeaf(i)
        q.put((freqs[i], leaf))
    while q.qsize() > 1:
        l1 = q.get()
        l2 = q.get()
        weight = l1[0] + l2[0]
        node = huffmanNode(l1[1], l2[1])
        q.put((weight,node))
    root = q.get()[1]
    huff_map = dict()
    root.makeMap(huff_map, '')
    return huff_map

import numpy as np
from scipy.signal import fftconvolve

# given the current approximation frame image, and the filters lo_d and hi_d
# initialize empty arrays
temp = np.zeros([image.shape[0], image.shape[1]/2])
LL = np.zeros([image.shape[0]/2, image.shape[1]/2])
LH = np.zeros([image.shape[0]/2, image.shape[1]/2])
HL = np.zeros([image.shape[0]/2, image.shape[1]/2])
HH = np.zeros([image.shape[0]/2, image.shape[1]/2])

# low-pass filtering along the rows
for i in xrange(image.shape[0]):
	temp[i] = fftconvolve(image[i], lo_d, mode='full')[1::2]

# low and hi-pass filtering along the columns
for i in xrange(image.shape[1]/2):
	LL[:,i] = fftconvolve(temp[:,i],lo_d,mode='full')[1::2]
    LH[:,i] = fftconvolve(temp[:,i],hi_d,mode='full')[1::2]

# hi-pass filtering along the rows
for i in xrange(image.shape[0]):
	temp[i] = fftconvolve(image[i], hi_d, mode='full')[1::2]

# low and hi-pass filtering along the columns
for i in xrange(image.shape[1]/2):
	HL[:,i] = fftconvolve(temp[:,i],lo_d,mode='full')[1::2]
    HH[:,i] = fftconvolve(temp[:,i],hi_d,mode='full')[1::2]

# given current coefficients LL, LH, HL, HH
# initialize temporary arrays
n = LL.shape[0]
temp1 = np.zeros([2*n,n])
temp2 = np.zeros([2*n,n])
up1 = np.zeros(2*n)
up2 = np.zeros(2*n)

# upsample and filter the columns of the coefficient arrays
for i in xrange(n):
	up1[1::2] = HH[:,i]
	up2[1::2] = HL[:,i]
	temp1[:,i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]
	up1[1::2] = LH[:,i]
	up2[1::2] = LL[:,i]
	temp2[:,i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]

# upsample and filter the rows, then add results together
result = sp.zeros([2*n,2*n])
for i in xrange(2*n):
	up1[1::2] = temp1[i]
	up2[1::2] = temp2[i]
	result[i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]

>>> # calculate one level of wavelet coefficients
>>> coeffs = pywt.wavedec2(lena,'haar', level=1)
