from re import X
import numpy as np
import cmath
import scipy.stats as st
# from calculateFeatureOptions import *

# from RealityModel import *

FINITE_MAX = np.finfo(np.float32).max
FINITE_SMALLEST_DIGIT = np.finfo(np.float64).eps

# feature extraction function should only take one input(data).
"""
Required data format is same as for train function.
For multichannel data, extractFeatures extraction
is performed seperately on each channel,
then the channels are concatenated horizontally.
"""


# to add more extractFeaturess, update getfeat to include a dict entry
# with key = extractFeatures name and value = a function taking a window
# and producing the extractFeatures extracted version
# extractFeatures extracted version of a window is a single row consisting
# of extractFeatures extraction performed on each channel followed by
# horizontal concatenation
# getfeat = dict()
# TODO:
# getfeat['C5'] = lambda wind:
#               (np.abs(fft.rfft(wind,axis=0))/
#                np.sqrt(np.sum(np.abs(fft.rfft(wind,axis=0)),
#                axis = 0))).flatten('F')
# output is a (r p ) x n array where r is the size of the extractFeatures
# vector of each channel, p is number of channels, n is number
# of windows
# feats = np.stack([getfeat[self.extractFeatures](wind) for wind in data])

# unitilty functions
# change +/-inf to FINITE_MAX
# change nan values to zero\

def check_output_edge(values):
    out = np.nan_to_num(values, nan=0, posinf=FINITE_MAX, neginf=-1 * FINITE_MAX)
    return np.array(out)


def A1(data, **kwargs):
    return data.flatten('F')


# !!!!!! Warning!!!!!! In this version, Python A2 is not matching the matlab result in case data<0 !!!!!! Warning!!!!!!
# Taking the ABS for all the inputs
def A2(data, **kwargs):
    data[data == 0] = FINITE_SMALLEST_DIGIT
    output = np.log(np.abs(data))
    return np.array(output).flatten('F')


def A3(data, **kwargs):
    return np.square(data).flatten('F')


def A4(data, **kwargs):
    return np.diff(data, axis=0).flatten('F')


def A5(data, **kwargs):
    return np.sign(data).flatten('F')


def A6(data, **kwargs):
    return np.sign(np.absolute(np.diff(np.sign(data), axis=0))).flatten('F')


def C1(data, **kwargs):
    spectralmag = np.abs(np.fft.rfft(data, axis=0))[0:int(np.ceil(data.shape[0] / 2))]
    return spectralmag.flatten('F')


def C2(data, **kwargs):
    p = np.absolute(C1(data) ** 2)
    p[p == 0] = FINITE_SMALLEST_DIGIT
    return np.log(p)


def C3(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    spectrum = np.fft.fft(data, axis=0)
    phase = np.angle(spectrum)
    unwrapped = np.unwrap(phase, axis=0)
    samples = phase.shape[0]
    center = (samples + 1) // 2
    if samples == 1:
        center = 0
    ndelay = np.array(np.round(unwrapped[center] / np.pi))
    unwrapped -= (np.pi * ndelay / center) * (np.arange(samples).reshape(samples, 1))
    spectrum[spectrum == 0] = FINITE_SMALLEST_DIGIT
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped
    ceps = np.fft.ifft(log_spectrum, axis=0).real
    return ceps.flatten('F')


def C4(data, **kwargs):
    complex_rfft = np.fft.rfft(data, axis=0)[0:int(np.ceil(data.shape[0] / 2))]
    return np.angle(complex_rfft).flatten('F')


def C5(data, **kwargs):
    spectralmag = np.abs(np.fft.rfft(data, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(data.shape[0] / 2))]
    d = np.sqrt(np.sum(spectralmag ** 2, axis=0))
    if np.isscalar(d):
        if d == 0: d = 1
    else:
        d[d == 0] = 1
    normspectralmag = spectralmag / d
    flattened = normspectralmag.flatten('F')
    return flattened


# !!!!!! Warning!!!!!! Python C6 is using check_output_edge() its ok to use in this case since the length of filter_bank is fixed 40 !!!!!! Warning!!!!!!
# !!!!!! Warning!!!!!! If you decide to use any filter number much larger than 40, please note C6 can be very slow                   !!!!!! Warning!!!!!!
def C6(data, **kwargs):
    complex = np.fft.rfft(data, axis=0)[:-1]
    real = np.real(complex)
    imag = np.imag(complex)
    mag = np.sqrt(np.square(real) + np.square(imag))
    pow_mag = (mag ** 2) / (len(mag) / 2)

    # If dont have a correct mel-filter-bank, throw this message
    if 'MelFilterBank' in kwargs:
        MelFilterBank = np.array(kwargs['MelFilterBank'])
    else:
        print(
            '!!!!!!featureExtraction.py Warning!!!!!! You dont have a proper Mel-filter-Bank, check your sample rate or MelfiterBank in feature options.......!!!!!!featureExtraction.py Warning!!!!!!')
        MelFilterBank = np.ones((40, len(pow_mag)))

    filter_banks = []
    for i in range(len(MelFilterBank)):
        filter_banks.append(np.dot(MelFilterBank[i], pow_mag))
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    WeightedMag = 20 * np.log10(filter_banks)
    output = WeightedMag.flatten('F')
    return check_output_edge(output)


# !!!!!! Warning!!!!!! Python BA1 is using check_output_edge() its ok to use in this case since the length of this output is fixed 18 !!!!!! Warning!!!!!!
def BA1(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    var = np.var(data, ddof=1, axis=0)
    components = np.array([
        np.mean(data, axis=0),  # Mean
        np.std(data, ddof=1, axis=0),  # Standard Deviation
        np.sqrt(np.mean(data ** 2, axis=0)),  # RMS energy
        st.moment(data, moment=3, axis=0) / (var ** 1.5),  # Skewness
        st.moment(data, moment=4, axis=0) / (var ** 2),  # fisher kurtosis, -3 if using Pearson's
        -np.sum((data ** 2) * np.log(data ** 2), axis=0),  # Entropy
        st.moment(data, moment=5, axis=0),  # Moment 5
        st.moment(data, moment=6, axis=0),  # Moment 6
        st.moment(data, moment=7, axis=0),  # Moment 7
        st.iqr(data, axis=0),  # Interquartile range
        (np.max(data, axis=0) - np.min(data, axis=0)),  # Range
        var,  # Variance
        np.prod(np.abs(data), axis=0) ** (1.0 / data.shape[0]),  # Geometric mean
        data.shape[0] / np.sum(1.0 / data, axis=0),  # Harmonic mean
        np.median(data, axis=0),  # Median
        st.mode(data, axis=0)[0][0],  # Mode
        np.max(data, axis=0),  # Maximum
        np.min(data, axis=0)  # Minimum
    ])
    components = components.flatten('F')
    if 'FeatureSubset' in kwargs:
        filter = np.array(kwargs['FeatureSubset'])
        return check_output_edge(components[filter])
    else:
        return check_output_edge(components)


# !!!!!! Warning!!!!!! In this version, Python A2 is not matching the matlab result in case data<0 !!!!!! Warning!!!!!!
# Taking the ABS for all the inputs
def BA2(data, **kwargs):
    data[data == 0] = FINITE_SMALLEST_DIGIT
    output = np.log(np.abs(data))
    return BA1(output, **kwargs)


def BA3(data, **kwargs):
    return BA1(np.square(data), **kwargs)


def BA4(data, **kwargs):
    return BA1(np.diff(data, axis=0), **kwargs)


def BA5(data, **kwargs):
    return BA1(np.sign(data), **kwargs)


def BA6(data, **kwargs):
    return BA1(np.sign(np.absolute(np.diff(np.sign(data), axis=0))), **kwargs)


def BC1(data, **kwargs):
    return BA1(np.abs(np.fft.rfft(data, axis=0))[0:int(np.ceil(data.shape[0] / 2))], **kwargs)


def BC2(data, **kwargs):
    spectralmag = np.abs(np.fft.rfft(data, axis=0))[0:int(np.ceil(data.shape[0] / 2))]
    p = np.absolute(spectralmag ** 2)
    p[p == 0] = FINITE_SMALLEST_DIGIT
    return BA1(np.log(p), **kwargs)


def BC3(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    spectrum = np.fft.fft(data, axis=0)
    phase = np.angle(spectrum)
    unwrapped = np.unwrap(phase, axis=0)
    samples = phase.shape[0]
    center = (samples + 1) // 2
    if samples == 1:
        center = 0
    ndelay = np.array(np.round(unwrapped[center] / np.pi))
    unwrapped -= (np.pi * ndelay / center) * (np.arange(samples).reshape(samples, 1))
    spectrum[spectrum == 0] = FINITE_SMALLEST_DIGIT
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped
    ceps = np.fft.ifft(log_spectrum, axis=0).real
    return BA1(ceps, **kwargs)


def BC4(data, **kwargs):
    complex_rfft = np.fft.rfft(data, axis=0)[0:int(np.ceil(data.shape[0] / 2))]
    return BA1(np.angle(complex_rfft), **kwargs)


def BC5(data, **kwargs):
    spectralmag = np.abs(np.fft.rfft(data, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(data.shape[0] / 2))]
    d = np.sqrt(np.sum(spectralmag ** 2, axis=0))
    if np.isscalar(d):
        if d == 0: d = 1
    else:
        d[d == 0] = 1
    normspectralmag = spectralmag / d
    return BA1(normspectralmag, **kwargs)


def BC6(data, **kwargs):
    MelFilterBank = kwargs['MelFilterBank']
    complex = np.fft.rfft(data, axis=0)[:-1]
    real = np.real(complex)
    imag = np.imag(complex)
    mag = np.sqrt(np.square(real) + np.square(imag))
    pow_mag = (mag ** 2) / (len(mag) / 2)
    filter_banks = []
    for i in range(len(MelFilterBank)):
        filter_banks.append(np.dot(MelFilterBank[i], pow_mag))
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    WeightedMag = 20 * np.log10(filter_banks)
    return BA1(WeightedMag, **kwargs)


def SubWindow(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if 'SubWindowSize' not in kwargs:
        swlen = int(2 * 2 ** np.ceil(np.log2(np.sqrt(len(data)))))
    else:
        swlen = kwargs['SubWindowSize']

    if 'SubStepSize' not in kwargs:
        sstep = int(swlen / 2)
    else:
        sstep = kwargs['SubStepSize']
    sw = np.dstack([data[i:(i + swlen)]
                    for i in range(0, len(data), sstep)
                    if (i + swlen) <= len(data)])
    return sw


def E1C1(data, **kwargs):
    subwindow = SubWindow(data)
    spectralmag = np.abs(np.fft.rfft(subwindow, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(subwindow.shape[0] / 2))]
    f = spectralmag
    return np.swapaxes(f, 1, 2).flatten('F').real


def E1C2(data, **kwargs):
    subwindow = SubWindow(data)
    swlen = len(subwindow)
    nfft = int(swlen)
    fftlen = int(np.ceil(nfft / 2))
    fftsw = np.fft.fft(subwindow, n=nfft, axis=0)[0:fftlen, :, :]
    log_no_inf = np.vectorize(lambda x: FINITE_SMALLEST_DIGIT if x == 0 else np.log(x))
    log = log_no_inf(np.square(np.abs(fftsw)))
    return np.swapaxes(log, 1, 2).flatten('F').real


def E1C3(data, **kwargs):
    subwindow = SubWindow(data)
    spectrum = np.fft.fft(subwindow, axis=0)
    phase = np.angle(spectrum)
    unwrapped = np.unwrap(phase, axis=0)
    samples = phase.shape[0]
    center = (samples + 1) // 2
    if samples == 1:
        center = 0
    ndelay = np.array(np.round(unwrapped[center] / np.pi))
    unwrapped -= (np.pi * ndelay / center) * (np.arange(samples)[:, None, None])
    spectrum[spectrum == 0] = FINITE_SMALLEST_DIGIT
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped
    ceps = np.fft.ifft(log_spectrum, axis=0)
    return np.swapaxes(ceps, 1, 2).flatten('F').real


def E1C4(data, **kwargs):
    subwindow = SubWindow(data)
    complex_rfft = np.fft.rfft(subwindow, axis=0)
    complex_rfft = complex_rfft[0:int(np.ceil(subwindow.shape[0] / 2))]
    a = np.angle(complex_rfft)
    return np.swapaxes(a, 1, 2).flatten('F').real


def E1C5(data, **kwargs):
    subwindow = SubWindow(data)
    spectralmag = np.abs(np.fft.fft(subwindow, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(subwindow.shape[0] / 2))]
    d = np.sqrt(np.sum(spectralmag ** 2, axis=0))
    if np.isscalar(d):
        if d == 0: d = 1
    else:
        d[d == 0] = 1
    normspectralmag = spectralmag / d
    flattened = np.swapaxes(normspectralmag, 1, 2).flatten('F')
    return flattened


# E1C6 is not working for now......................
def E1C6(data, **kwargs):
    subwindow = SubWindow(data)
    MelFilterBank = kwargs['MelFilterBank']
    complex = np.fft.rfft(subwindow, axis=0)[:-1]
    real = np.real(complex)
    imag = np.imag(complex)
    mag = np.sqrt(np.square(real) + np.square(imag))
    pow_mag = (mag ** 2) / (len(mag) / 2)
    filter_banks = []
    for i in range(len(MelFilterBank)):
        filter_banks.append(np.dot(MelFilterBank[i], pow_mag))
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    WeightedMag = 20 * np.log10(filter_banks)
    flattened = np.swapaxes(WeightedMag, 1, 2).flatten('F')
    return check_output_edge(flattened)


def E2C1(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if 'SubWindowSize' not in kwargs:
        swlen = int(2 * 2 ** np.ceil(np.log2(np.sqrt(len(data)))))
    else:
        swlen = kwargs['SubWindowSize']
    if 'NFFT2' not in kwargs:
        nfft2 = int(swlen / 2)
    else:
        nfft2 = kwargs['NFFT2']
    if 'FFTLEN2' not in kwargs:
        fftlen2 = int(np.ceil(nfft2 / 2))
    else:
        fftlen2 = kwargs['FFTLEN2']
    if 'SubStepSize' not in kwargs:
        sstep = int(np.round(swlen / 2))
    else:
        sstep = kwargs['SubStepSize']

    sw = np.dstack([data[i:(i + swlen)]
                    for i in range(0, len(data), sstep)
                    if (i + swlen) <= len(data)])
    spectralmag = np.abs(np.fft.rfft(sw, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(sw.shape[0] / 2))]
    f = spectralmag
    f2 = np.abs(np.fft.fft(f, n=nfft2, axis=2)[:, :, 0:fftlen2])
    return np.swapaxes(f2, 0, 1).flatten('A').real


def E2C2(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if 'SubWindowSize' not in kwargs:
        swlen = int(2 * 2 ** np.ceil(np.log2(np.sqrt(len(data)))))
    else:
        swlen = kwargs['SubWindowSize']
    if 'NFFT' not in kwargs:
        nfft = int(swlen)
    else:
        nfft = kwargs['NFFT2']
    if 'FFTLEN' not in kwargs:
        fftlen = int(np.ceil(nfft / 2))
    else:
        fftlen = kwargs['FFTLEN']
    if 'NFFT2' not in kwargs:
        nfft2 = int(swlen / 2)
    else:
        nfft2 = kwargs['NFFT2']
    if 'FFTLEN2' not in kwargs:
        fftlen2 = int(np.ceil(nfft2 / 2))
    else:
        fftlen2 = kwargs['FFTLEN2']
    if 'SubStepSize' not in kwargs:
        sstep = int(np.round(swlen / 2))
    else:
        sstep = kwargs['SubStepSize']

    sw = np.dstack([data[i:(i + swlen)]
                    for i in range(0, len(data), sstep)
                    if (i + swlen) <= len(data)])
    fftsw = np.fft.fft(sw, n=nfft, axis=0)[0:fftlen, :, :]
    log_no_inf = np.vectorize(lambda x: -1e15 if x == 0 else np.log(x))
    f = log_no_inf(np.square(np.abs(fftsw)))
    f2 = np.abs(np.fft.fft(f, n=nfft2, axis=2)[:, :, 0:fftlen2])
    return np.swapaxes(f2, 0, 1).flatten('A').real


def E2C4(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if 'SubWindowSize' not in kwargs:
        swlen = int(2 * 2 ** np.ceil(np.log2(np.sqrt(len(data)))))
    else:
        swlen = kwargs['SubWindowSize']
    if 'NFFT' not in kwargs:
        nfft = int(swlen)
    else:
        nfft = kwargs['NFFT2']
    if 'FFTLEN' not in kwargs:
        fftlen = int(np.ceil(nfft / 2))
    else:
        fftlen = kwargs['FFTLEN']
    if 'NFFT2' not in kwargs:
        nfft2 = int(swlen / 2)
    else:
        nfft2 = kwargs['NFFT2']
    if 'FFTLEN2' not in kwargs:
        fftlen2 = int(np.ceil(nfft2 / 2))
    else:
        fftlen2 = kwargs['FFTLEN2']
    if 'SubStepSize' not in kwargs:
        sstep = int(np.round(swlen / 2))
    else:
        sstep = kwargs['SubStepSize']
    sw = np.dstack([data[i:(i + swlen)]
                    for i in range(0, len(data), sstep)
                    if (i + swlen) <= len(data)])

    complex_rfft = np.fft.rfft(sw, axis=0)
    complex_rfft = complex_rfft[0:int(np.ceil(sw.shape[0] / 2))]
    angle = np.angle(complex_rfft)
    fftsw = np.abs(np.fft.fft(angle, n=nfft2, axis=-1)[:, :, 0:fftlen2])
    return np.swapaxes(fftsw, 0, 1).flatten('A').real


def E2C5(data, **kwargs):
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if 'SubWindowSize' not in kwargs:
        swlen = int(2 * 2 ** np.ceil(np.log2(np.sqrt(len(data)))))
    else:
        swlen = kwargs['SubWindowSize']
    if 'NFFT2' not in kwargs:
        nfft2 = int(swlen / 2)
    else:
        nfft2 = kwargs['NFFT2']
    if 'FFTLEN2' not in kwargs:
        fftlen2 = int(np.ceil(nfft2 / 2))
    else:
        fftlen2 = kwargs['FFTLEN2']
    if 'SubStepSize' not in kwargs:
        sstep = int(np.round(swlen / 2))
    else:
        sstep = kwargs['SubStepSize']

    sw = np.dstack([data[i:(i + swlen)]
                    for i in range(0, len(data), sstep)
                    if (i + swlen) <= len(data)])
    spectralmag = np.abs(np.fft.rfft(sw, axis=0))
    spectralmag = spectralmag[0:int(np.ceil(sw.shape[0] / 2))]
    d = np.sqrt(np.sum(spectralmag ** 2, axis=0))
    if np.isscalar(d):
        if d == 0: d = 1
    else:
        d[d == 0] = 1
    normspectralmag = spectralmag / d
    f = normspectralmag
    f2 = np.abs(np.fft.fft(f, n=nfft2, axis=2)[:, :, 0:fftlen2])
    return np.swapaxes(f2, 0, 1).flatten('A').real


def magnitude(data):
    return np.linalg.norm(data, axis=1).reshape(-1, 1)


def appendMagnitude(data):
    return np.concatenate((data, magnitude(data)), axis=1)


class combineFeatures():
    def __init__(self, featList):
        self.featList = featList

    def __call__(self, data):
        return np.hstack([feat(data) for feat in self.featList])


class composeFeatures():
    def __init__(self, featList):
        self.featList = featList

    def __call__(self, data):
        out = data
        for func in reversed(self.featList):
            out = func(out)
        return out


class sensorComposite():
    # sensorChannels is a list of lists of 0s or 1s like
    # [[1,1,0],[1,1],[0,0,0]
    # 1 if that channel is supposed to be on, 0 otherwise
    # ith list is for ith sensor
    # featSpaces is a list representing the feature space for each sensor
    def __init__(self, sensorChannels, featSpaces, featOptList=None):
        if featOptList is None:
            self.featOptList = [{} for f in featSpaces]
        else:
            self.featOptList = featOptList
        self.sensorChannels = sensorChannels
        self.featSpaces = featSpaces

    def __call__(self, data):
        # this chooses the appropriate columns for each sensor and takes correspondign feat space of them
        idx = np.cumsum([0] + [len(sens) for sens in self.sensorChannels])
        out = np.hstack([
            self.featSpaces[i](data[:, idx[i]:idx[i + 1]][:,
                               np.array(self.sensorChannels[i]).astype(np.bool)],
                               **self.featOptList[i])
            for i in range(len(idx) - 1)])
        return out


def GetChannels(data):
    return data.T


def passthrough(data):
    return data


def Flatten(data):
    return data.flatten()


def toArray(data):
    return np.array(data)