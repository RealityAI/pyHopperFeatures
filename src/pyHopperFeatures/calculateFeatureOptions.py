import numpy as np

#calculate MelFilterBank, may not needed if the mel filter bank will be take from json file
#Keep the ablity of mel filter bank calculations for python now, may need this later
def computeMelFilterBank(sample_rate, winLen, nfilt=40):
        NFFT = winLen
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        #FFTBinNums = np.ceil(hz_points*(NFFT+1) / sample_rate)
        FreqResHz = (sample_rate / 2) / np.ceil(NFFT/2)
        FFTBinNums = hz_points / FreqResHz
        FFTBinNums = np.ceil(FFTBinNums)
        FFTBinNums = FFTBinNums + 1

        bin = np.minimum(FFTBinNums, np.ceil(NFFT/2)*np.ones(len(FFTBinNums)))
        fbank = np.zeros((nfilt, int(np.ceil(NFFT / 2 ))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
            for k in range(f_m_minus,f_m_plus+1):
                if f_m_minus<= k <f_m:
                    fbank[m - 1, k-1] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                elif f_m< k <=f_m_plus:
                    fbank[m - 1, k-1] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
                else:
                    fbank[m - 1, k-1] = 1
        return fbank

def findClosestVal(sample_rate, winLen, Fval):
    NFFT = winLen
    HzBins = np.linspace(0, int(sample_rate / 2), int(np.ceil(NFFT/2)))
    idx = (np.abs(HzBins - Fval)).argmin()
    return idx

def featureSubset(startIdx,stopIdx,nfft):
    endPoint = int(np.ceil(nfft/2))
    idx = np.ones((endPoint),dtype=int)
    idx[:startIdx] = 0
    idx[stopIdx:] = 0
    return idx

def window_data(rawdata,
                winlen,
                winstep):
    """
    Utility function for creating windowed data from raw data
    Output is in required format of train function.
    """
    rawdata = np.float32(rawdata)
    windows = [rawdata[t:(t + winlen)]
               for t in range(0, len(rawdata), winstep)
               if (t + winlen) <= len(rawdata)]

    return windows