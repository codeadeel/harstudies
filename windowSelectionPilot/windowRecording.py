# This file is responsible for cutting one SONAR recording into windows with their labels, recognizer features, scores and CPU time
# %%
# Importing Libraries
import time

import numpy as np


# %%
# Window Recording
class windowRecording:
    def loadRecording(self, recordingName):
        """
        This method loads the signal, time stamp and label arrays of one converted recording

        Arguments
        =========
        recordingName : Recording name as in recordingIndex.csv

        Output
        ======
        Tuple of the signal array, the time stamps and the sample labels
        """
        # Load The Recording Arrays
        signalArray = np.load(self.recordingDirectory / f"{recordingName}_signals.npy")
        sampleTimes = np.load(self.recordingDirectory / f"{recordingName}_time.npy")
        sampleLabels = np.load(self.recordingDirectory / f"{recordingName}_labels.npy")
        return signalArray, sampleTimes, sampleLabels

    def dropRepeatedRows(self, signalArray, sampleTimes, sampleLabels, sampleCounts):
        """
        This method drops the rows that repeat the time stamp of the row before and counts them

        Arguments
        =========
        signalArray : Samples by channels array of the recording
        sampleTimes : Microsecond time stamps of the recording
        sampleLabels : Label code of every sample
        sampleCounts : Dictionary of sample counts, its repeatedRows entry is set

        Output
        ======
        Tuple of the signal array, the time stamps and the sample labels without the repeated rows
        """
        # Drop Rows That Repeat The Time Stamp Of The Row Before
        newTimes = np.concatenate([[True], np.diff(sampleTimes) > 0])
        sampleCounts["repeatedRows"] = int((~newTimes).sum())
        return signalArray[newTimes], sampleTimes[newTimes], sampleLabels[newTimes]

    def cutStreams(self, signalArray, sampleTimes, sampleLabels, sampleCounts):
        """
        This method cuts every stream of a recording into whole windows after filling its missing values

        Arguments
        =========
        signalArray : Samples by channels array of the recording
        sampleTimes : Microsecond time stamps of the recording
        sampleLabels : Label code of every sample
        sampleCounts : Dictionary of sample counts, updated while the streams are cut

        Output
        ======
        Tuple of the lists of window signals, window sample labels and stream numbers, one entry per kept stream
        """
        # Cut Every Stream Into Whole Windows After Filling Its Missing Values
        windowParts, labelParts, streamParts = [], [], []
        for firstSample, endSample in self.splitStreams(sampleTimes):
            streamSignals, filledCount = self.holdLastValid(signalArray[firstSample:endSample].astype(np.float64))
            windowCount = (endSample - firstSample) // self.windowSamples
            if streamSignals is None:
                sampleCounts["droppedStreams"] += 1
                sampleCounts["droppedStreamSamples"] += endSample - firstSample
                continue
            sampleCounts["filledValues"] += filledCount
            if windowCount == 0:
                continue
            windowedSamples = windowCount * self.windowSamples
            windowParts.append(streamSignals[:windowedSamples].reshape(windowCount, self.windowSamples, -1))
            labelParts.append(sampleLabels[firstSample:firstSample + windowedSamples].reshape(windowCount, self.windowSamples))
            streamParts.append(np.full(windowCount, sampleCounts["streams"]))
            sampleCounts["streams"] += 1
            sampleCounts["windowedSamples"] += windowedSamples
        return windowParts, labelParts, streamParts

    def emptyRecording(self, recordingName, sampleCounts):
        """
        This method returns the result of a recording that has no whole window

        Arguments
        =========
        recordingName : Recording name as in recordingIndex.csv
        sampleCounts : Dictionary of sample counts of the recording

        Output
        ======
        Dictionary of empty window arrays, zero CPU seconds and the sample counts
        """
        # Return Empty Arrays And Zero Times
        return {
            "recording": recordingName, "features": np.zeros((0, len(self.featureNames)), dtype=np.float32), "scores": np.zeros((0, len(self.scoreNames))),
            "labels": np.zeros(0, dtype=np.uint8), "streams": np.zeros(0, dtype=np.int64), "cpuSeconds": dict.fromkeys(["features", *self.scoreNames], 0.0), "counts": sampleCounts,
        }

    def scoreWindows(self, windowSignals, streamStarts):
        """
        This method computes the recognizer features and each score of the windows, timing each in CPU seconds

        Arguments
        =========
        windowSignals : Windows by samples by channels array
        streamStarts : Boolean array marking the first window of every stream

        Output
        ======
        Tuple of the feature matrix, the list of score columns and the dictionary of CPU seconds
        """
        # Compute The Recognizer Features
        cpuSeconds = {}
        cpuStart = time.process_time()
        featureMatrix = self.recognizerFeatures(windowSignals)
        cpuSeconds["features"] = time.process_time() - cpuStart

        # Compute Each Score
        scoreColumns = []
        for scoreName in self.scoreNames:
            cpuStart = time.process_time()
            if scoreName == "windowChange":
                scoreColumns.append(self.scoreWindowChange(windowSignals, streamStarts))
            elif scoreName == "accelerationVariance":
                scoreColumns.append(self.scoreAccelerationVariance(windowSignals))
            else:
                scoreColumns.append(self.scoreSpectralEntropy(windowSignals))
            cpuSeconds[scoreName] = time.process_time() - cpuStart
        return featureMatrix, scoreColumns, cpuSeconds

    def buildRecording(self, recordingName):
        """
        This method cuts one recording into windows and computes their labels, recognizer features and scores, timing each computation in CPU seconds

        Arguments
        =========
        recordingName : Recording name as in recordingIndex.csv

        Output
        ======
        Dictionary of window arrays, CPU seconds and sample counts
        """
        # Load The Recording Arrays
        signalArray, sampleTimes, sampleLabels = self.loadRecording(recordingName)
        sampleCounts = {"samples": len(signalArray), "repeatedRows": 0, "windowedSamples": 0, "streams": 0, "filledValues": 0, "droppedStreams": 0, "droppedStreamSamples": 0}

        # Drop Rows That Repeat The Time Stamp Of The Row Before
        signalArray, sampleTimes, sampleLabels = self.dropRepeatedRows(signalArray, sampleTimes, sampleLabels, sampleCounts)

        # Cut Every Stream Into Whole Windows After Filling Its Missing Values
        windowParts, labelParts, streamParts = self.cutStreams(signalArray, sampleTimes, sampleLabels, sampleCounts)

        # Stop Early For A Recording Without A Whole Window
        if not windowParts:
            return self.emptyRecording(recordingName, sampleCounts)
        windowSignals = np.concatenate(windowParts)
        streamNumbers = np.concatenate(streamParts)
        streamStarts = np.concatenate([[True], streamNumbers[1:] != streamNumbers[:-1]])

        # Compute The Recognizer Features And Each Score, Timing Each In CPU Seconds
        featureMatrix, scoreColumns, cpuSeconds = self.scoreWindows(windowSignals, streamStarts)
        return {
            "recording": recordingName,
            "features": featureMatrix,
            "scores": np.column_stack(scoreColumns),
            "labels": self.majorityLabels(np.concatenate(labelParts)),
            "streams": streamNumbers,
            "cpuSeconds": cpuSeconds,
            "counts": sampleCounts,
        }
