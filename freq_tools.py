import numpy as np
import warnings
from scipy import signal
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import scipy.signal as ss
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def generateFFT(inputData,samplingRate,nPerSeg,averaged=True):
    ''' Function to get the FFT for a response
    #
    # Adapted from Scipy's Welch's Method function:
    # https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/spectral.py#L286-L427
    #
    # Input:
    #   inputData - reference signal we wish to analyze
    #   minResolution - The minimum distance between two spectral peaks
    #   pointsBetween - The number of bins between the minimal spectral peaks
    #   maxSegments - The maximum number of segments to divide the data into
    #   averaged - Dictates whether the various segments should be averaged together
    #
    # Output:
    #   fftFreq = an array of the freqs used in the FFT
    #   fftMag = an array of the amplitude of the FFT at each freq in fftFreq
    #
    ########################################################################
    '''

    inputData = np.atleast_2d(inputData)

    NyquistFreq = 0.5 * samplingRate

    # Force the overlap to be one-half of the window length
    nOverlap = nPerSeg // 2

    # This code breaks the input data into a set of segments
    step = nPerSeg - nOverlap
    shape = inputData.shape[:-1]\
            + ((inputData.shape[-1]-nOverlap)//step, nPerSeg)
    strides = inputData.strides[:-1]\
            + (step*inputData.strides[-1], inputData.strides[-1])

    inputData = np.lib.stride_tricks.as_strided(inputData, shape=shape,
                                            strides=strides)

    # Create a hanning window
    window = np.hanning(nPerSeg)

    # Create the windowed FFT magnitudes
    result = inputData - np.expand_dims(np.mean(inputData,axis=-1),axis=-1)
    result = window * result
    FFTMag = np.fft.rfft(result, n=nPerSeg)
    FFTMag = np.conjugate(FFTMag) * FFTMag

    # Do stuff to scale the magnitudes.
    # Review the original source for insight.
    if nPerSeg % 2:
        result[..., 1:] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        result[..., 1:-1] *= 2

    scale = 1.0 / (samplingRate * (window*window).sum())
    FFTMag *= scale

    FFTMag = np.rollaxis(FFTMag, -1, -2)

    # Get the frequencies for the FFT
    FFTFreq = np.fft.rfftfreq(nPerSeg, 1/samplingRate)

    if averaged:
        FFTMag = np.mean(FFTMag,axis=-1).flatten()
    else:
        FFTMag = FFTMag[0,:,:]

    return FFTFreq.flatten(), FFTMag.real, result

def generateRawFFT(samplingRate, signalData):
    """
    DO NOT USE : this is just to prove Georgia Pacific we are able to deal with
    Fourier transforms

    (float) 1 * n array signalData : temporal data of the signal
    (float) timestep : time between two temporal data points of the signal
    """
    amps = np.fft.fft(signalData)
    nbFreqs = len(amps)
    freqs = np.fft.fftfreq(nbFreqs, d= 1 / samplingRate)

    # processing the data to get rid of complex imaginary part
    amps = np.abs(amps[ 2: nbFreqs // 2])# getting rid of the 2 first indexes which is common practice
    freqs = freqs[ 2: nbFreqs //2]

    return(freqs, amps)

def demodulation(samplingRate, rawData, passBand, lowPass):
    """
    (float) samplingRate
    (float array) 1 * n waveFormData : temporal data of the signal, ie :
                                    datapoints of the waveForm
    (float) lowFreq : below this frequency, all the amplitudes will be forced
                      to 0
    (float) highFreq : above this frequency, all amplitudes will be forced to
                       0

        return:

        (float array) 1 * n freqs : array of the spectrum's frequencies
        (float array) 1 * n newAmpsMod : array of the spectrum's amplitudes
    """
    # Reference : Vibration-Based Condition Monitoring, Robert Bond Randall, page 201

    # Making sure the waveform data is in the right format
    rawData = rawData.flatten()

    rawData -= np.mean(rawData)

    NyquistFreq = 0.5 * samplingRate

    lowCut, highCut = passBand

    #creating the filter
    b,a = signal.butter(3,[lowCut / NyquistFreq, highCut / NyquistFreq], btype='bandpass', output='ba')

    #applying the filter to the signal
    filteredSignal = signal.filtfilt(b, a, rawData)
    rectifiedSignal = np.abs(filteredSignal)

    num, den = signal.butter(3, lowPass / NyquistFreq, btype='lowpass')

    demodulatedSignal = signal.filtfilt(num, den, rectifiedSignal)

    demodulatedSignal -= np.mean(demodulatedSignal)

    return demodulatedSignal

def spectralKurtosis(samplingRate, waveFormData, nperseg = 256):
    # Reference : Vibration-Based Condition Monitoring, Robert Bond Randall, page 172
    # getting the evolution of the spectrum over time
    """
    (float) samplingRate
    (1 * k) waveFormData : vibration data of the coming from the accelerometer
    (int) neperseg : number of points per time segment for the Short Time fourier
                    Transform

    returns

     (1 * n float) fSTFT : array of sample frequencies
     (1 * m float) tSTFT : array of segment times
     (n * m float) stftxx : Short Time Fourier Transform of waveFormData
    """

    fSTFT, tSTFT, stftxx = signal.stft(waveFormData, fs = samplingRate, nperseg = nperseg)

    fourthPowerMean = np.sum(np.abs(stftxx) ** 4, axis = 1) / len(tSTFT)

    # taking some liberties with the original formula : the next commented line
    # should be the right formula whereas the next uncommented line is a Modified
    # formula. But it seems it's working much better whith the modified one
    #squareMeanSquare = (np.sum(np.abs(stftxx) ** 2, axis = 1) / len(tSTFT)) ** 2
    squareMeanSquare = (np.sum(np.abs(stftxx) ** 2) / len(tSTFT)) ** 2

    spectKurto = fourthPowerMean / squareMeanSquare - 2

    return(fSTFT, tSTFT, stftxx, spectKurto)


def kurtogram(samplingRate, waveFormData, npersegMin = 32, npersegMax = 256):
    # Reference : Vibration-Based Condition Monitoring, Robert Bond Randall, page 176
    """
    (float) samplingRate
    (1 * k) waveFormData : vibration data of the coming from the accelerometer
    (int) nepersegMin : number minimum of points per time segment for the Short
                        Time fourier Transform
    (int) nepersegMax : number maximum of points per time segment for the Short
                        Time fourier Transform

    returns

     (float) freqFiltering : Frequency around which the waveFormData should
                             be bandpassed
    """

    freqFiltering = 0
    skMax = float('-inf')

    rang = np.linspace(npersegMin, npersegMax + 1, 100, dtype = int)

    for k in rang:
        fSTFT, tSTFT, stftxx, spectKurto = spectralKurtosis(samplingRate, waveFormData, nperseg = k)

        n = len(spectKurto)

        # taking the max of a slice of the spectral Kurtosis to avoid the edge effects
        if n < 60:
            maxi = float(np.max(spectKurto[3 : -3]))
        else:
            maxi = float(np.max(spectKurto[int(0.05 * n) : int(0.95 * n)]))

        if maxi > skMax:
            skMax = maxi
            freqFiltering = fSTFT[np.argwhere(spectKurto == skMax)]

    return(float(freqFiltering))

def meda(signal, sizeFilter = 30):
    """
    reference : Multipoint Optimal Minimum Entropy Deconvolutionand Convolution Fix:
    Application to vibration fault detectionGeoff L. McDonalda Qing Zhao

    MEDA algorithm used to get a signal closer from the original one

    (1 * n float) signal : waveForm data
    (int) sizeFilter : size of the filter used to get the original signal back

    returns :
    (1 * n float) filtered signal
    (1 * m float) coefficient of the filter
    """
    sig = signal.flatten()

    # Step 1 : initialize filter
    filt = np.zeros((sizeFilter))
    filt[sizeFilter // 2] = 1
    filt[sizeFilter // 2 + 1] =  - 1
    filtNew = np.ones(sizeFilter)

    # Step 2 : Calculate X0 and inv(X0 * X0.T) from input signal x
    nbPts = len(sig)
    X0 = np.zeros((sizeFilter, nbPts - sizeFilter + 1))

    # filling the X0 array with the appropriate values (making it a upper diagonal)
    for k in range(sizeFilter):
            X0[k, k : ] = sig[np.arange(sizeFilter - 1, nbPts - k)]
            X0[k, : k ] = sig[np.arange(sizeFilter - k - 1, sizeFilter  - 1)]


    while np.sum(np.abs(filtNew - filt)) > 0.001:
        filt = filtNew
        k+=1
        # Step 3: Calculate y as X0.T * f
        y = np.dot(X0.T, filt)

        # Step 4 : finding the new coefficients of the filter
        coefficient = np.sum(y[ : nbPts - sizeFilter + 1] ** 2) / np.sum(y[ : nbPts - sizeFilter + 1] ** 4)
        matrix = np.linalg.solve(np.dot(X0,X0.T), X0)
        filtNew = coefficient * matrix
        filtNew = np.dot(filtNew, (y[ : nbPts - sizeFilter + 1].T) ** 3)

        # normalizing the  filter result
        filtNew = filtNew / (np.sum(filtNew ** 2)) ** 0.5

    originalSig = np.dot(X0.T, filtNew)

    return(originalSig, filtNew)


def getPeakFreqs(rawFFTFreq, rawFFTMag, broadBandPct=50,stdThreshold=10):
    '''
    broadBandPct - Percentile below which the FFT magnitude is
                   assumed to be broadband noise.
    stdThreshold - number of standard deviations to add to the
                   median, setting the "signal" threshold.
    '''

    # Reset the arrays
    peakFreqs = np.array([])
    peakMags = np.array([])

    inputSignal = rawFFTMag.flatten()

    # Assume that the
    lowerStd = np.std(
                inputSignal[np.argwhere(
                            inputSignal < np.percentile(inputSignal,broadBandPct))])

    upperThreshold = stdThreshold * lowerStd + np.median(inputSignal)

    # Consider candidate locations where we can safely reject
    # the null hypothesis
    candidateLocs = np.argwhere((rawFFTMag > upperThreshold)).flatten()

    if candidateLocs.size < 1:
        raise ValueError('A signal cannot be derived from the supplied\
                        \nfrequency data. It is likely that your input\
                        \ndata is too noisy.')

    # We want sequential frequency regions where the null
    # hypothesis is rejected.
    candidateFreqs = np.split(candidateLocs,
        np.where(np.diff(candidateLocs) != 1)[0]+1)

    if len(candidateFreqs) == 1:
        warnings.warn('Only one peak was detected. If this returns\
                     \nunexpected results, consider increasing the\
                     \nupper threshold used for peak detection. If\
                     \nthis warning follows a "low resolution"\
                     \nwarning, it is likely that the input data is\
                     \ninsufficient to resolve the desired\
                     \nfrequencies.')

    # For each region where we rejected the null hypothesis
    for i in range(len(candidateFreqs)):
        if candidateFreqs[i].shape[0] > 1:
            # Get the weighted sum to find a nominal frequency
            # within this region
            #nominalFreq = np.sum(rawFFTMag[candidateFreqs[i]] \
            #            * rawFFTFreq[candidateFreqs[i]]) \
            #            / np.sum(rawFFTMag[candidateFreqs[i]])
            indexMax = np.argmax(rawFFTMag[candidateFreqs[i]]).flatten()[0]
            nominalFreq = candidateFreqs[i][indexMax]
            # Add these frequencies and magnitudes
            # to the peaks
            peakFreqs = np.append(peakFreqs,rawFFTFreq[nominalFreq])
            #peakMags = np.append(
            #                    peakMags,
            #                    np.mean(rawFFTMag[candidateFreqs[i]]))
            peakMags = np.append(peakMags, rawFFTMag[candidateFreqs[i]][indexMax])

    return(peakFreqs, peakMags)

def getActualHz(inputRPM, rawSignal, samplingRate, percentageBand = 0.1):
    '''
    Search in a band of frequencies around an approximate machine
    operating speed to find the 'true' operating speed,
    based on frequency-domain analysis
    (float) percentageBand : used to create a band centered around the inputRPM
    freq where we are going to look for the real RPM of the signal
    ([(1 - percentageBand) * inputRPMFreq, (1 + percentageBand) * inputRPMFreq])
    '''
    rawFFTFreq, rawFFTMag = generateRawFFT(rawSignal, samplingRate)

    # get the index of the inputRPM in the spectrum
    inputHz = inputRPM / 60.
    resolutionFreq = rawFFTFreq[-1] / rawFFTFreq.shape[-1] # interval between the different frequency points
    indexHz = int(round(inputHz / resolutionFreq, 0))

    # range in which we are going to look for the RPM speed
    lowIndex = int(indexHz - round(percentageBand * inputHz / resolutionFreq))
    highIndex = int(indexHz + round(percentageBand * inputHz / resolutionFreq) + 1) # to take into account that the [low : high] list in python
    #in fact represents numbers from low, low + 1, ..., high - 1

    if lowIndex < 0 :
        lowIndex = 0
    if highIndex > rawFFTFreq.shape[-1] :
        highIndex = rawFFTFreq.shape[-1] - 1

    copyArray = rawFFTMag[lowIndex : highIndex]

    ampliMax = np.max(rawFFTMag[lowIndex : highIndex])
    actualIndexHz = np.argwhere(copyArray == ampliMax)

    # Extract the indices of rawFFTFreq where the frequency
    # is withing the specified tolerance of the inputRPM
    #searchIndices = np.argwhere(
    #    np.abs(rawFFTFreq - inputHz) / inputHz \
    #    < self.freqTolerance)
#
#        if len(peakFreqs) == 0:
#                              real spindle speed.')
#                              raise ValueError('No peak frequency detected. Cannot retrieve\n\
#
#        if len(searchIndices) == 0:
#            raise ValueError('The given spindle speed does not correspond\n\
#                           with any frequency in the FFT. Please ensure\n\
#                           the correct spindle speed is given.\n\
#                           Input Speed: {}\n\
#                           Frequency Peaks: {}\n'.format(inputHz,self.peakFreqs))

    # The actual spindle speed is given by the frequency location
    # where the FFT magnitude is greatest within the search indices
#        actualHz = rawFFTFreq[
#                    searchIndices[np.argmax(rawFFTMag[searchIndices])]]

    return float(rawFFTFreq[actualIndexHz + lowIndex])