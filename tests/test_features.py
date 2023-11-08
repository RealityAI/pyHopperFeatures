import unittest
import sys
import os
import h5py
import numpy as np


sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)), ".."))


# from RealityModel import *
# from algos import *
from src.featureExtraction import *
from src.calculateFeatureOptions import *

path = os.path.dirname(os.path.abspath(__file__))


# class TestRealityModel(unittest.TestCase):
#     def setUp(self):
#         self.mod = RealityModel(DefaultModel(),
#                                 C5,
#                                 3,
#                                 2)
#         pass
#
#     def test_window(self):
#         mod = self.mod
#         data = np.array([1, 2, 3, 4, 5])
#         self.assertTrue(np.all(
#             mod.window([1, 2, 3, 4, 5], 2) ==
#             np.array([[1, 2, 3], [3, 4, 5]]))
#         )
#
#     def test_train(self):
#         np.random.seed(0)
#         mod = self.mod
#         mod.train([np.random.rand(3) for i in range(100)])
#         preds = mod.predict([1, 2, 3])
#         self.assertEqual(preds,
#                          [[1.029281265593971, 'ANOMALY'],
#                           [6.804460551533802, 'ANOMALY'],
#                           [12.802153959537488, 'ANOMALY']])


class DiagnosticCheck:
    def __init__(self):
        self.feList = None
        self.windowedData = None
        self.opt = None
        self.feListDict = None

    def setUp(self):

        # Feature with error range <E-05
        self.feListDict = {
            'A1': A1, 'A3': A3, 'A4': A4, 'A5': A5, 'A6': A6,
            'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5, 'C6': C6,
            'E1C1': E1C1, 'E1C2': E1C2, 'E1C3': E1C3, 'E1C4': E1C4, 'E1C5': E1C5,
            'E2C1': E2C1, 'E2C4': E2C4, 'E2C5': E2C5
        }
        """
        self.feListDict = {
            'BA1':BA1,'BA3':BA3,'BA4':BA4,'BA5':BA5,'BA6':BA6,
            'BC1':BC1,'BC2':BC2,'BC3':BC3,'BC4':BC4,'BC5':BC5,'BC6':BC6,
        }
        """
        # set up feature options
        self.opt = {
            'sampleRate': 8000,
            # 'FValStart':200,
            # 'FValStop':1200
        }

        # load and window raw data
        winLen = 256
        step = 128
        raw = np.genfromtxt(path + '/matlab_files/3channel_raw.csv', delimiter=',')
        windowed = window_data(raw, winLen, step)
        self.windowedData = windowed

        # list of features for unit test
        self.feList = self.feListDict.keys()

        # pre-calculating feature options
        self.opt['MelFilterBank'] = computeMelFilterBank(self.opt['sampleRate'], winLen)
        pass

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
            # print(f"feature space {feSpace} passed")
            print("passed")

if __name__ == '__main__':
    diag_check = DiagnosticCheck()
    diag_check.setUp()
    diag_check.test_fe_data()