# This file is responsible for preparing the recordings and handing out their cached windows and features
# %%
# Importing Libraries
import numpy as np

from activityRecognitionChain.slidingWindows import slidingWindows


# %%
# Recording Access
class recordingAccess:
    def replaceFilledFrames(self, dataMatrix, fillReplacement):
        """
        This method replaces the filled frames of every IMU with values computed from its unfilled frames, without using labels

        Arguments
        =========
        dataMatrix : Data matrix ( frames , channels ) as published
        fillReplacement : mean for the IMU's per channel mean over its unfilled frames, interpolate for a straight line between the nearest unfilled frames with the edge values held

        Output
        ======
        Copy of the data matrix with the filled frames replaced
        """
        # Find The Filled Frames Of Every IMU
        replacedMatrix = dataMatrix.copy()
        filledFrames = self.loader.findFilledFrames(dataMatrix)
        frameIndices = np.arange(len(dataMatrix))
        for imuIndex, imuNumber in enumerate(self.loader.placementNames):
            imuChannels = self.loader.selectChannels([f"acc_{imuNumber}", f"gyr_{imuNumber}"])
            imuFilled = filledFrames[:, imuIndex]
            if not imuFilled.any():
                continue

            # Use The Channel Means Or Interpolate Every Channel Between The Unfilled Frames
            imuData = dataMatrix[:, imuChannels]
            if fillReplacement == "mean":
                replacedMatrix[np.ix_(imuFilled, imuChannels)] = imuData[~imuFilled].mean(axis=0)
            elif fillReplacement == "interpolate":
                replacedMatrix[np.ix_(imuFilled, imuChannels)] = np.column_stack([
                    np.interp(frameIndices[imuFilled], frameIndices[~imuFilled], imuData[~imuFilled, channelPosition])
                    for channelPosition in range(len(imuChannels))
                ])
            else:
                raise ValueError(f"Unknown fill replacement {fillReplacement}")
        return replacedMatrix

    def secondsToSamples(self, durationSeconds):
        """
        This method converts a duration to a whole number of frames, at least one

        Arguments
        =========
        durationSeconds : Duration in seconds

        Output
        ======
        Number of frames
        """
        # Round To The Nearest Frame
        return max(1, int(round(durationSeconds * self.loader.samplingRate)))

    def getWindows(self, subjectNumber, windowSamples, stepSamples):
        """
        This method returns the cached windows of one recording with their labels and repetition blocks

        Arguments
        =========
        subjectNumber : Participant number
        windowSamples : Window length in frames
        stepSamples : Step in frames

        Output
        ======
        Tuple of the slidingWindows object, the window labels and the repetition block of every window centre
        """
        # Build The Windows Once
        cacheKey = (subjectNumber, windowSamples, stepSamples)
        if cacheKey not in self.windowCache:
            recording = self.recordings[subjectNumber]
            windows = slidingWindows(len(recording["labels"]), windowSamples, stepSamples)
            self.windowCache[cacheKey] = (
                windows, windows.labelWindows(recording["labels"], self.classLabels), recording["repetitions"][windows.centreFrames],
            )
        return self.windowCache[cacheKey]

    def getFeatures(self, subjectNumber, windowSamples, stepSamples, featureType):
        """
        This method returns the cached features of one recording for every channel

        Arguments
        =========
        subjectNumber : Participant number
        windowSamples : Window length in frames
        stepSamples : Step in frames
        featureType : One of Raw, VerySimple, Simple, FFT and All

        Output
        ======
        Tuple of the feature matrix, the feature names and the channel index of every feature
        """
        # Extract The Features Once
        cacheKey = (subjectNumber, windowSamples, stepSamples, featureType)
        if cacheKey not in self.featureCache:
            recording = self.recordings[subjectNumber]
            windows, _, _ = self.getWindows(subjectNumber, windowSamples, stepSamples)
            self.featureCache[cacheKey] = self.extractor.extractFeatures(
                recording["data"], windows, featureType, self.loader.channelNames, recording["channelCentres"],
            )
        return self.featureCache[cacheKey]
