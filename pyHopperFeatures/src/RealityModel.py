import time
import sys
import os
from multiprocessing import Process, Queue

RealityModelScriptAbsPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(RealityModelScriptAbsPath, "..", "reality-infrastructure"))
import numpy as np
from numpy import fft as fft
import pickle
import featureExtraction
from algos import *
from network import *
import smoothing

import time
import os
import logging

logger = logging.getLogger('RAI')


def deleteOld(outdir):
    allfiles = os.listdir(outdir)
    for file in allfiles:
        filetime = float(file.replace(".csv", ""))
        if filetime + 60 * 60 < time.time():  # 1 hour
            os.remove(os.path.join(outdir, file))


def archiver(sensorParams):
    while (1):
        allfiles = os.listdir(sensorParams["outdir"])
        for file in allfiles:
            filetime = float(file.replace(".csv", ""))
            if filetime > time.time() + 60 * 60:  # 1 hour
                os.remove(os.path.join(sensorParams["outdir"], file))


class RealityModel:
    # reality models contain:
    # an algo object ex. SVM
    # a feature extraction scheme (function)
    # a window length
    # a channel count
    # a default step (for windowing data)
    # (may genericize to "shape" in future)

    # TIME Dimension (when present) should always be along axis 0
    def __new__(cls, *args, **kwargs):
        out = super(RealityModel, cls).__new__(cls)
        out.featureOptions = {}
        return out

    def __init__(self,
                 algoObject,
                 extractFeaturesFunction,
                 windowLength,
                 defaultStep,
                 smooth=None,
                 sensorName=None,
                 featureOptions=None):

        self.algo = algoObject
        # (algo object muct contain fit() and predict() functions)
        self.train = algoObject.fit
        self.updateFit = algoObject.updateFit
        self.predict = algoObject.predict
        # extractfeatures expects windowed data
        self.extractFeatures = extractFeaturesFunction
        self.windowLength = windowLength
        self.defaultStep = int(defaultStep)
        self.smooth = smooth if not smooth == None else smoothing.passthrough()
        self.filename = "notFromFile.pkl"
        self.reportIntervalSeconds = 0
        self.pushPort = 60001  # set True if reporting to port 6000
        self.sensorName = sensorName
        self.featureOptions = {} if featureOptions is None else featureOptions

        self.count = 0

    def window(self, data, step):
        # break a continuous data array into a list of chunks
        return RealityModel.window_data(data, self.windowLength, step)

    # a feature extraction scheme
    def fe_and_train(self, windows):
        """
        Input data should be a list of windows, where
        each window is a nxp numpy array.
        p i the number of channels, and
        n is the window length.
        extractFeatures extraction is done automatically.
        """
        features = []
        for window in windows:
            features += [self.extractFeatures(window, **self.featureOptions)]
        self.train(features)

    def train_on_files(self, filelist):
        """
        Input data should be a list of files
        extractFeatures extraction is done automatically.
        """

        windows = []
        for file in filelist:
            # read the data from file
            filedata = pickle.load(open(file, "rb"))[1]
            # window it, and add to big list
            windows += [self.window(filedata, self.defaultStep)]

        self.fe_and_train(windows)

    def train_on_directory(self, directory):
        filelist = []
        for file in os.listdir(directory):
            if file.endswith(".bin"):
                filelist += [file]

        self.train_on_files(filelist)

    def fe_and_predict(self, windows):
        """
        Required data format is same as for train function.
        Return a list of predictions
        """
        features = []
        for window in windows:
            features += [self.extractFeatures(window, **self.featureOptions)]

        return self.predict(features)

    def predict_on_file(self, file):
        """
        Input data should be a list of files
        extractFeatures extraction is done automatically.
        """

        metadata, data = pickle.load(open(file, "rb"))
        windows = self.window(data, self.defaultStep)
        return self.fe_and_predict(windows)

    def window_and_fe(self, data):
        """
        Input data should be Txp where T is number of time points
        and p is number of channels.
        This is a utility function that first performs windowing on
        raw data and then executes train function.
        """
        features = []
        for window in self.window(data, self.defaultStep):
            features += [self.extractFeatures(window, **self.featureOptions)]
        return features

    def window_and_fe_and_train(self, data):
        """
        Input data should be Txp where T is number of time points
        and p is number of channels.
        This is a utility function that first performs windowing on
        raw data and then executes train function.
        """
        self.fe_and_train([w for w in self.window(data, self.defaultStep)])

    def fe_and_update_fit(self, windows):
        """
        Input data should be a list of windows, where
        each window is a nxp numpy array.
        p i the number of channels, and
        n is the window length.
        extractFeatures extraction is done automatically.
        """
        features = []
        for window in windows:
            features += [self.extractFeatures(window, **self.featureOptions)]
        self.updateFit(features)

    def window_and_fe_and_update_fit(self, data):
        self.fe_and_update_fit([w for w in self.window(data, self.defaultStep)])

    def train_on_data(self, data):
        self.window_and_fe_and_train(data)

    def train_on_new_data(self, data):
        self.window_and_fe_and_update_fit(data)

    def window_and_fe_and_predict(self, data):
        """
        Similar to window_and_fe_and_train.
        """
        return self.fe_and_predict(self.window(data, self.defaultStep))

    def window_data(rawdata,
                    winlen,
                    winstep):
        """
        Utility function for creating windowed data from raw data
        Output is in required format of train function.
        """
        rawdata = np.float32(rawdata)
        logger.debug(winlen)
        windows = [rawdata[t:(t + winlen)]
                   for t in range(0, len(rawdata), winstep)
                   if (t + winlen) <= len(rawdata)]

        return windows

    def toFile(self, filename):
        pickle.dump(self, open(filename, "wb+"))

    def inferOnDataSmoothing(self, data):
        predictions = self.window_and_fe_and_predict(data)
        smoothedPredictions = self.smooth.update(predictions)
        return smoothedPredictions

    def writeData(writecontent, outfilename):
        np.savetxt(outfilename, writecontent, delimiter=',')

    def runInference(self, inports, reportTimeInterval,
                     record=True):  # for now, only allow 1 inport (single sensor model)

        dataSubscription = subscribeSensorData(inports[0])
        inferencePusher = getPusher(self.pushPort)

        lastReport = 0

        datadir = os.path.join(os.path.dirname(self.filename), os.path.basename(self.filename) + "_data")
        os.makedirs(datadir, exist_ok=True)

        # get some starter data
        pickledData = dataSubscription.recv()
        params, rawdata = pickle.loads(pickledData)

        # modeldict["reportIntervalSeconds"]
        while (1):
            pickledData = dataSubscription.recv()
            params, y = pickle.loads(pickledData)
            rawdata = np.append(rawdata, y, axis=0)
            logger.debug(np.shape(rawdata))

            reportInterval = int(reportTimeInterval) or self.reportIntervalSeconds
            if time.time() > lastReport + reportInterval and len(rawdata) >= self.windowLength:
                lastReport = time.time()
                # run the model on it
                prediction = self.inferOnDataSmoothing(rawdata)
                # logger.debug(str(prediction))

                # SEND IT ALL - MGMT
                # write to file
                currTime = str(time.time())
                # outfilename should be modelfile dirname / modelfile_data
                # outfilename = os.path.join(datadir, currTime + ".pkl")
                outfilename = os.path.join(datadir, currTime + ".csv")
                if record:
                    logger.debug("writing to file " + outfilename)
                    # with open(outfilename, "wb") as file:
                    #    file.write(pickle.dumps([params, rawdata]))

                    # write stuff to file
                    Process(target=RealityModel.writeData, args=(rawdata, outfilename)).start()

                smoothingMethod = str(self.smooth)
                smoothingType = smoothingMethod[10:smoothingMethod.find('Smoothing')].capitalize()
                smoothingParamters = smoothingMethod[smoothingMethod.find('(') + 1:len(smoothingMethod) - 1].split(',')
                reportJson = dict()
                reportJson["state"] = [pred[1] for pred in prediction]
                reportJson["data"] = outfilename
                reportJson["timeStamp"] = currTime
                reportJson["score"] = [pred[0] for pred in prediction]
                reportJson["modelFilename"] = self.filename
                reportJson["threshold"] = self.algo.getThreshold()
                reportJson[
                    "smoothingMethod"] = smoothingType + ' Smoothing' if not 'Passthrough(' else 'No Smoothing method applied.'
                if (smoothingParamters[0] != ''):
                    reportJson["smoothingWindow"] = smoothingParamters[0]
                    if (smoothingType == 'Vote'):
                        reportJson["minVotes"] = smoothingParamters[2]
                        reportJson["selectedClass"] = smoothingParamters[3]

                logger.debug("sending inference")
                inferencePusher.send_string(json.dumps(reportJson))

                # keep buffer short
                rawdata = rawdata[len(self.window(rawdata, step=self.defaultStep)) * self.defaultStep:]

            # create a new file, delete old files
            deleteOld(datadir)


def modelFromHopper(filename):
    # convert a Hopper Json into a RealityModel
    pass


def modelFromFile(filename):
    a = pickle.load(open(filename, "rb"))
    a.filename = filename
    return a


def inferOnFilename(modelFilename, dataFilename):
    return modelFromFile(modelFilename).predict_on_file(dataFilename)


# add to class
def inferOnData(modelFilename, data):
    return modelFromFile(modelFilename).window_and_fe_and_predict(data)


if __name__ == "__main__":
    # upside down model, good for accelerometer testing
    from algos import *

    a = RealityModel(algoObject=upsideDown(), extractFeaturesFunction=featureExtraction.passthrough, windowLength=1,
                     channelCount=3, defaultStep=256)
    logger.debug(a.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/accOK.bin")))
    logger.debug(a.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/accUpsideDown.bin")))
    upFilename = os.path.join(RealityModelScriptAbsPath, "testModels/upsideDown.pkl")
    a.toFile(upFilename)

    b = modelFromFile(upFilename)
    logger.debug(b.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/accOK.bin")))
    logger.debug(b.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/accUpsideDown.bin")))

    a = RealityModel(algoObject=rms(), extractFeaturesFunction=featureExtraction.toArray, windowLength=100,
                     channelCount=1, defaultStep=256)
    upFilename = os.path.join(RealityModelScriptAbsPath, "testModels/loud.pkl")
    logger.debug(a.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/loud.bin")))
    logger.debug(a.predict_on_file(os.path.join(RealityModelScriptAbsPath, "testData/quiet.bin")))
    a.toFile(upFilename)

    # default model
    # b =  ADModel()
    # b.train(np.array([[1,1,1],[2,2,2],[3,3,3]]))
    # logger.debug(b.predict(np.array([[1,1,1],[2,2,2],[3,3,3]])))
    # b.toFile("defaultModel.pkl")