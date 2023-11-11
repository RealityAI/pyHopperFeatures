# Description: This file is used to run the unit test for feature extraction functions
import os
import sys
import h5py
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'src' directory (assuming 'hopper_controller.py' is inside 'pyHopper')
src_dir = os.path.abspath(os.path.join(script_dir, ".."))

print(src_dir)

# Add the 'src' directory to sys.path
sys.path.insert(0, src_dir)


# Get the path of the package directory (assuming main_script.py is in the same directory as my_package).
package_dir = os.path.dirname(os.path.abspath(__file__))

# Add the package directory to sys.path.
sys.path.insert(0, package_dir)

import os
from featureExtraction import *
from calculateFeatureOptions import *
path = os.path.dirname(os.path.abspath(__file__))

from featureExtraction import *

class DiagCheck:
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
            matTestPath = path+'/matlab_files/'+feSpace+'.mat'
            with h5py.File(matTestPath, 'r') as f:
                a_group_key = list(f.keys())[0]
                matOut = np.array(f[a_group_key])
            print('Python/Matlab output MAX difference in feature space: ' + feSpace +' is off by: '+ str(np.max(np.abs(pyFeData-matOut))))
            # assertTrue(lower<1+np.average((pyFeData)-matOut)<upper)


def main():
    dg = DiagCheck()
    dg.setUp()
    dg.test_fe_data()

if __name__ == '__main__':
    main()