# This file is responsible for describing the segments of both recordings and recording the dataset checks
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Data Description
class dataDescription:
    def writeDataInventory(self):
        """
        This method writes the class segments and durations of both recordings and records the dataset checks

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per participant and class
        """
        # Describe Every Class Of Every Recording
        loader = self.evaluator.loader
        inventoryRows = []
        blocksHoldEachGestureOnce = True
        for subjectNumber, recording in self.evaluator.recordings.items():
            inventoryRows += self.describeClasses(subjectNumber, recording)

            # Check That Every Repetition Block Holds Every Gesture Exactly Once
            blocksHoldEachGestureOnce &= self.checkRepetitionBlocks(recording)
            self.recordParameter("data", f"framesSubject{subjectNumber}", len(recording["labels"]))
            self.recordParameter("data", f"minutesSubject{subjectNumber}", len(recording["labels"]) / loader.samplingRate / 60)
            self.recordParameter("data", f"repetitionBlocksSubject{subjectNumber}", int(recording["repetitions"].max() + 1))
        inventoryTable = pd.DataFrame(inventoryRows)
        writeCsv(inventoryTable, self.resultsDirectory / "dataInventory.csv")

        # Record The Layout And The Block Check
        self.recordDataLayout(blocksHoldEachGestureOnce)
        return inventoryTable

    def describeClasses(self, subjectNumber, recording):
        """
        This method describes the segments and durations of every class in one recording

        Arguments
        =========
        subjectNumber : Participant number
        recording : Dictionary with the data, labels and repetitions of the participant

        Output
        ======
        List with one row per class
        """
        # Measure The Duration Of Every Segment
        loader = self.evaluator.loader
        segmentStarts, segmentEnds, segmentLabels = loader.findSegments(recording["labels"])
        segmentSeconds = (segmentEnds - segmentStarts) / loader.samplingRate

        # Describe The Segments Of Every Class
        inventoryRows = []
        for classIndex, classLabel in enumerate(loader.classLabels):
            classSeconds = segmentSeconds[segmentLabels == classLabel]
            inventoryRows.append({
                "subject": subjectNumber, "classLabel": classLabel, "className": loader.classNames[classIndex],
                "segments": len(classSeconds), "frames": int(np.sum(recording["labels"] == classLabel)),
                "seconds": float(np.sum(recording["labels"] == classLabel)) / loader.samplingRate,
                "minSegmentSeconds": classSeconds.min(), "medianSegmentSeconds": float(np.median(classSeconds)),
                "maxSegmentSeconds": classSeconds.max(),
                "shareOfSegmentsInPaperRange": float(np.mean((classSeconds >= self.paperActivitySeconds[0]) & (classSeconds <= self.paperActivitySeconds[1]))),
            })
        return inventoryRows

    def checkRepetitionBlocks(self, recording):
        """
        This method checks that every repetition block of a recording holds every gesture exactly once

        Arguments
        =========
        recording : Dictionary with the labels and repetitions of the participant

        Output
        ======
        True when every block holds every gesture exactly once
        """
        # Find The Segments And The Block Of Every Segment
        loader = self.evaluator.loader
        gestureLabels = [classLabel for classLabel in loader.classLabels if classLabel != loader.nullLabel]
        segmentStarts, _, segmentLabels = loader.findSegments(recording["labels"])
        segmentBlocks = recording["repetitions"][segmentStarts]

        # Compare The Gestures Of Every Block With The List Of Gestures
        holdsEachGestureOnce = True
        for blockNumber in np.unique(recording["repetitions"]):
            blockGestures = sorted(segmentLabels[(segmentBlocks == blockNumber) & (segmentLabels != loader.nullLabel)].tolist())
            holdsEachGestureOnce &= blockGestures == gestureLabels
        return holdsEachGestureOnce

    def recordDataLayout(self, blocksHoldEachGestureOnce):
        """
        This method records the channel layout, the classes and the result of the block check

        Arguments
        =========
        blocksHoldEachGestureOnce : Whether every block of both recordings holds every gesture exactly once

        Output
        ======
        None
        """
        # Record The Channel Layout, The Classes And The Block Check
        loader = self.evaluator.loader
        self.recordParameter("data", "channels", len(loader.channelNames))
        self.recordParameter("data", "channelNames", " ".join(loader.channelNames))
        self.recordParameter("data", "samplingRateHz", loader.samplingRate)
        self.recordParameter("data", "classes", len(loader.classNames))
        self.recordParameter("data", "classNames", "; ".join(f"{classLabel} {className}" for classLabel, className in zip(loader.classLabels, loader.classNames)))
        self.recordParameter("data", "placements", "; ".join(f"IMU {imuNumber} {placementName}" for imuNumber, placementName in loader.placementNames.items()))
        self.recordParameter("data", "everyBlockHoldsEachGestureOnce", bool(blocksHoldEachGestureOnce))
