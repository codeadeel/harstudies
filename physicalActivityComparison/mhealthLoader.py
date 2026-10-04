# This file is responsible for loading the MHEALTH recordings, keeping the nine accelerometer signals at 25 Hz and splitting the labelled samples into activity segments
# %%
# Importing Libraries
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate


# %%
# MHEALTH Loader
class mhealthLoader:
    def __init__(self, dataRoot):
        """
        This class initializes the loader with the MHEALTH folder, the accelerometer columns and the activity names of the dataset README

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets

        Output
        ======
        None
        """
        # Store The Data Location And Rates
        self.dataDirectory = Path(dataRoot) / "mhealth" / "MHEALTHDATASET"
        self.subjects = list(range(1, 11))
        self.sourceRate = 50.0
        self.decimationFactor = 2
        self.filterOrder = 8
        self.targetRate = self.sourceRate / self.decimationFactor

        # Store The Accelerometer Columns Of The README ( Zero Based ) And The Label Column
        self.placementColumns = {"chest": [0, 1, 2], "leftAnkle": [5, 6, 7], "rightLowerArm": [14, 15, 16]}
        self.signalNames = [f"{placementName}_{axisName}" for placementName in self.placementColumns for axisName in "xyz"]
        self.labelColumn = 23
        self.columnCount = 24
        self.nullLabel = 0

        # Store The Activity Names Of The README ( Label 0 Means No Activity )
        self.classNames = {
            1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down", 4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
            7: "Frontal elevation of arms", 8: "Knees bending", 9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front and back",
        }
        self.classLabels = list(self.classNames)

    def loadSubject(self, subjectNumber):
        """
        This method reads one recording and returns its nine accelerometer signals at 25 Hz with the matching labels

        Arguments
        =========
        subjectNumber : Participant number, 1 to 10

        Output
        ======
        Dictionary with the decimated signals ( samples , 9 ), their labels and the labels at the source rate
        """
        # Read The Tab Separated Log
        logTable = pd.read_csv(self.dataDirectory / f"mHealth_subject{subjectNumber}.log", sep="\t", header=None)
        if logTable.shape[1] != self.columnCount:
            raise ValueError(f"Unexpected column count {logTable.shape[1]} for subject {subjectNumber}")
        signalColumns = [columnIndex for placementColumns in self.placementColumns.values() for columnIndex in placementColumns]
        sourceSignals = logTable.iloc[:, signalColumns].to_numpy(dtype=np.float64)
        sourceLabels = logTable.iloc[:, self.labelColumn].to_numpy(dtype=np.int64)

        # Low Pass Filter Forward And Backward, Then Keep One Sample In Each Decimation Step With Its Label
        targetSignals = decimate(sourceSignals, self.decimationFactor, n=self.filterOrder, ftype="iir", axis=0, zero_phase=True)
        targetLabels = sourceLabels[::self.decimationFactor]
        return {"signals": targetSignals, "labels": targetLabels, "sourceLabels": sourceLabels}

    def findSegments(self, frameLabels):
        """
        This method finds the runs of one activity label, leaving out the samples without activity

        Arguments
        =========
        frameLabels : Label of every sample at 25 Hz

        Output
        ======
        List of ( start , end , label ) tuples with exclusive ends
        """
        # Split The Timeline Where The Label Changes And Drop The No Activity Runs
        changeIndices = np.flatnonzero(np.diff(frameLabels) != 0) + 1
        runStarts = np.concatenate([[0], changeIndices])
        runEnds = np.concatenate([changeIndices, [len(frameLabels)]])
        return [
            (int(runStart), int(runEnd), int(frameLabels[runStart]))
            for runStart, runEnd in zip(runStarts, runEnds) if frameLabels[runStart] != self.nullLabel
        ]

    def loadAll(self):
        """
        This method loads every recording with its activity segments

        Arguments
        =========
        None

        Output
        ======
        Dictionary from participant number to its recording dictionary
        """
        # Load Every Participant
        recordings = {}
        for subjectNumber in self.subjects:
            recording = self.loadSubject(subjectNumber)
            recording["segments"] = self.findSegments(recording["labels"])
            recordings[subjectNumber] = recording
        return recordings

    def sampleTable(self, recordings):
        """
        This method lists every labelled sample with its participant, position and activity segment

        Arguments
        =========
        recordings : Output of loadAll

        Output
        ======
        Tuple of the sample signals ( samples , 9 ) and a dataframe with subject, frame, label and segment per sample
        """
        # Collect The Samples Of Every Activity Segment In Time Order
        signalParts, rowParts = [], []
        segmentNumber = 0
        for subjectNumber, recording in recordings.items():
            for segmentStart, segmentEnd, segmentLabel in recording["segments"]:
                frameIndices = np.arange(segmentStart, segmentEnd)
                signalParts.append(recording["signals"][frameIndices])
                rowParts.append(pd.DataFrame({"subject": subjectNumber, "frame": frameIndices, "label": segmentLabel, "segment": segmentNumber}))
                segmentNumber += 1
        return np.concatenate(signalParts), pd.concat(rowParts, ignore_index=True)

    def subsampleRows(self, sampleRows, sampleFraction, randomSeed):
        """
        This method draws the same seeded share of samples from every participant and class, keeping the time order

        Arguments
        =========
        sampleRows : Sample table of sampleTable
        sampleFraction : Share of the samples kept per participant and class
        randomSeed : Seed of the draw

        Output
        ======
        Sorted row positions of the kept samples
        """
        # Draw Without Replacement Inside Every Participant And Class
        randomGenerator = np.random.default_rng(randomSeed)
        keptRows = []
        for _, groupRows in sampleRows.groupby(["subject", "label"], sort=True):
            keptCount = max(1, int(round(sampleFraction * len(groupRows))))
            keptRows.append(randomGenerator.choice(groupRows.index.to_numpy(), size=keptCount, replace=False))
        return np.sort(np.concatenate(keptRows))
