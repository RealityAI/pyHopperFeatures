import unittest
import sys
import os
import h5py
import numpy as np
from src.pyHopperFeatures.featureExtraction import *
from src.pyHopperFeatures.calculateFeatureOptions import *

class TestFeatureSpace(unittest.TestCase):
    def setUp(self):
        self.feList = None
        self.windowedData = None
        self.opt = None
        self.feListDict = None

        # Feature with error range <E-05
        self.feListDict = {
            'A1': A1, 'A3': A3, 'A4': A4, 'A5': A5, 'A6': A6,
            'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5, 'C6': C6,
            'E1C1': E1C1, 'E1C2': E1C2, 'E1C3': E1C3, 'E1C4': E1C4, 'E1C5': E1C5,
            'E2C1': E2C1, 'E2C4': E2C4, 'E2C5': E2C5
        }

        # set up feature options
        self.opt = {
            'sampleRate': 8000,
            # 'FValStart':200,
            # 'FValStop':1200
        }

        # load and window raw data
        winLen = 256
        step = 128
        path = os.path.dirname(os.path.abspath(__file__))
        raw = np.genfromtxt(path + '/matlab_files/3channel_raw.csv', delimiter=',')
        windowed = window_data(raw, winLen, step)
        self.windowedData = windowed

        # list of features for unit test
        self.feList = self.feListDict.keys()

        # pre-calculating feature options
        self.opt['MelFilterBank'] = computeMelFilterBank(self.opt['sampleRate'], winLen)

    def test_fe_data(self):
        # test all the feature space functions in the dict
        for feSpace in self.feList:
            print("testing feature space " + feSpace + '........')
            func = self.feListDict[feSpace]
            feOpt = self.opt

            # set up py/mat error range
            lower = 0.999
            upper = 1.001

            # allow larger error range for B features
            if 'B' in feSpace:
                lower = 0.99
                upper = 1.01
            pyFeData = [func(window, **feOpt) for window in self.windowedData]
            # Add your assertions here to compare pyFeData with expected results.

if __name__ == '__main__':
    unittest.main()
