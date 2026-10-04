# This file is responsible for loading the ActRecTut gesture recordings with their channel names, class names and repetition blocks
# %%
# Importing Libraries
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# %%
# Gesture Loader
class gestureLoader:
    def __init__(self, dataRoot, samplingRate=32.0):
        """
        This class initializes the gesture loader with the ActRecTut data folder and the constants of the toolbox settings.m

        Arguments
        =========
        dataRoot : Root directory that holds the ActRecTut clone
        samplingRate : Joint sampling rate in hertz stated by the paper and settings.m ( default : 32.0 )

        Output
        ======
        None
        """
        # Store The Data Location And Rate
        self.dataDirectory = Path(dataRoot) / "ActRecTut" / "Data"
        self.samplingRate = samplingRate
        self.subjects = [1, 2]

        # Store The Channel Layout Of settings.m ( Three Accelerometer And Two Gyroscope Axes Per IMU )
        self.channelNames = [
            f"{modalityName}_{imuNumber}_{axisName}"
            for imuNumber in (1, 2, 3)
            for modalityName, axisNames in (("acc", "xyz"), ("gyr", "xy"))
            for axisName in axisNames
        ]
        self.placementNames = {1: "hand", 2: "lowerArm", 3: "upperArm"}

        # Store The Class Names In Label Order ( settings.m CLASSLABELS, Label 1 Is NULL )
        self.classNames = [
            "NULL", "Open window", "Drink", "Water plant", "Close window", "Cut", "Chop", "Stir", "Book",
            "Forehand", "Backhand", "Smash",
        ]
        self.nullLabel = 1
        self.classLabels = list(range(1, len(self.classNames) + 1))

    def loadSubject(self, subjectNumber):
        """
        This method loads the data matrix and the frame labels of one participant

        Arguments
        =========
        subjectNumber : Participant number, 1 or 2

        Output
        ======
        Tuple of the data matrix ( frames , channels ) and the integer label of every frame
        """
        # Read The Matlab File
        matlabContent = loadmat(self.dataDirectory / f"subject{subjectNumber}_gesture" / "data.mat")
        dataMatrix = np.asarray(matlabContent["data"], dtype=np.float64)
        frameLabels = np.asarray(matlabContent["labels"]).ravel().astype(np.int64)

        # Check The Layout Against The Toolbox Settings
        if dataMatrix.shape[1] != len(self.channelNames) or len(frameLabels) != len(dataMatrix):
            raise ValueError(f"Unexpected layout for subject {subjectNumber}: data {dataMatrix.shape}, labels {frameLabels.shape}")
        return dataMatrix, frameLabels

    def findSegments(self, frameLabels):
        """
        This method splits a label sequence into runs of identical labels

        Arguments
        =========
        frameLabels : Integer label of every frame

        Output
        ======
        Tuple of segment start frames, segment end frames ( exclusive ) and segment labels
        """
        # Find Where The Label Changes
        changeFrames = np.flatnonzero(np.diff(frameLabels) != 0) + 1
        segmentStarts = np.concatenate([[0], changeFrames])
        segmentEnds = np.concatenate([changeFrames, [len(frameLabels)]])
        return segmentStarts, segmentEnds, frameLabels[segmentStarts]

    def labelRepetitions(self, frameLabels):
        """
        This method numbers the repetition blocks, each running from the NULL segment before one occurrence of the first gesture to the next such NULL segment

        Arguments
        =========
        frameLabels : Integer label of every frame

        Output
        ======
        Zero based repetition number of every frame
        """
        # Find Every Occurrence Of The First Gesture Of The Cycle
        segmentStarts, _, segmentLabels = self.findSegments(frameLabels)
        firstGesture = segmentLabels[segmentLabels != self.nullLabel][0]
        cycleSegments = np.flatnonzero(segmentLabels == firstGesture)

        # Start Each Block At The NULL Segment Before That Occurrence
        blockStarts = []
        for segmentIndex in cycleSegments:
            precededByNull = segmentIndex > 0 and segmentLabels[segmentIndex - 1] == self.nullLabel
            blockStarts.append(segmentStarts[segmentIndex - 1] if precededByNull else segmentStarts[segmentIndex])
        blockStarts[0] = 0

        # Number Every Frame By The Last Block Start At Or Before It
        return np.searchsorted(np.asarray(blockStarts), np.arange(len(frameLabels)), side="right") - 1

    def selectChannels(self, sensorKeys):
        """
        This method returns the column indices of the requested sensors, for example acc_1 or gyr_3

        Arguments
        =========
        sensorKeys : Sequence of sensor keys in the settings.m style ( modality and IMU number )

        Output
        ======
        Sorted list of the matching channel column indices
        """
        # Match Every Axis Of Each Requested Sensor
        return [
            channelIndex for channelIndex, channelName in enumerate(self.channelNames)
            if channelName.rsplit("_", 1)[0] in sensorKeys
        ]

    def findFilledFrames(self, dataMatrix):
        """
        This method marks the frames whose IMU channels are all fractional, the sign of values filled in after recording ( raw sensor values are whole numbers )

        Arguments
        =========
        dataMatrix : Data matrix ( frames , channels )

        Output
        ======
        Boolean array ( frames , IMUs ), true where every channel of that IMU is fractional
        """
        # Test Every Channel Of Each IMU For A Fractional Part
        return np.stack([
            np.all(dataMatrix[:, self.selectChannels([f"acc_{imuNumber}", f"gyr_{imuNumber}"])] % 1 != 0, axis=1)
            for imuNumber in self.placementNames
        ], axis=1)
