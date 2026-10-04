# This file is responsible for loading the UCI HAR feature windows and raw inertial signals in one aligned row order
# %%
# Importing Libraries
from pathlib import Path

import numpy as np
import pandas as pd

from smartphoneExploration.harBouts import recordingBouts


# %%
# UCI HAR Loader, With The Recording Bout Methods Inherited From recordingBouts
class harLoader(recordingBouts):
    def __init__(self, dataRoot):
        """
        This class initializes the UCI HAR loader with the extracted dataset directory and the recording constants

        Arguments
        =========
        dataRoot : Root directory that holds the extracted uciHar folder

        Output
        ======
        None
        """
        # Store The Dataset Location
        self.datasetDirectory = Path(dataRoot) / "uciHar" / "UCI HAR Dataset"
        self.splitNames = {"train": "Train", "test": "Test"}

        # Store The Recording Constants Given In The Dataset Readme
        self.samplingRate = 50.0
        self.windowSamples = 128
        self.windowStepSamples = 64
        self.windowSeconds = self.windowSamples / self.samplingRate
        self.windowStepSeconds = self.windowStepSamples / self.samplingRate

    def loadFeatureNames(self):
        """
        This method reads features.txt and renames repeated names deterministically in file order

        Arguments
        =========
        None

        Output
        ======
        Tuple of the unique feature names and a dataframe mapping every original name to its unique name
        """
        # Read The Original Names In File Order
        featureLines = (self.datasetDirectory / "features.txt").read_text().strip().splitlines()
        originalNames = [featureLine.split(" ", 1)[1].strip() for featureLine in featureLines]

        # Rename Repeats The Way pandas Renames Duplicate Columns ( name, name.1, name.2 )
        usedNames = set()
        occurrenceCounts = {}
        mappingRows = []
        for featureIndex, originalName in enumerate(originalNames, start=1):
            occurrence = occurrenceCounts.get(originalName, 0)
            occurrenceCounts[originalName] = occurrence + 1
            suffixNumber = occurrence
            uniqueName = originalName if suffixNumber == 0 else f"{originalName}.{suffixNumber}"
            while uniqueName in usedNames:
                suffixNumber += 1
                uniqueName = f"{originalName}.{suffixNumber}"
            usedNames.add(uniqueName)
            mappingRows.append({
                "featureIndex": featureIndex,
                "originalName": originalName,
                "uniqueName": uniqueName,
                "occurrence": occurrence + 1,
                "renamed": uniqueName != originalName,
            })

        # Add The Total Occurrence Count Of Each Name
        featureMapping = pd.DataFrame(mappingRows)
        featureMapping.insert(4, "nameCount", featureMapping["originalName"].map(occurrenceCounts))
        return featureMapping["uniqueName"].tolist(), featureMapping

    def loadActivityLabels(self):
        """
        This method reads activity_labels.txt

        Arguments
        =========
        None

        Output
        ======
        Dictionary from activity label number to activity name, in label order
        """
        # Read The Label Table
        labelLines = (self.datasetDirectory / "activity_labels.txt").read_text().strip().splitlines()
        activityLabels = {int(labelLine.split(" ", 1)[0]): labelLine.split(" ", 1)[1].strip() for labelLine in labelLines}
        return dict(sorted(activityLabels.items()))

    def loadSplitFrame(self, splitName, featureNames, activityLabels):
        """
        This method loads the feature windows, subjects and activities of one split

        Arguments
        =========
        splitName : Split folder name, train or test
        featureNames : Unique feature names used as column names
        activityLabels : Dictionary from activity label number to activity name

        Output
        ======
        Dataframe with one row per window, the feature columns, subject, Activity and Data
        """
        # Read The Feature Windows
        splitDirectory = self.datasetDirectory / splitName
        windowFrame = pd.read_csv(splitDirectory / f"X_{splitName}.txt", sep=r"\s+", header=None, names=featureNames, dtype=np.float64)

        # Read The Subjects And Activity Labels
        subjects = np.loadtxt(splitDirectory / f"subject_{splitName}.txt", dtype=np.int64)
        labels = np.loadtxt(splitDirectory / f"y_{splitName}.txt", dtype=np.int64)
        if not len(windowFrame) == len(subjects) == len(labels):
            raise ValueError(f"Row counts differ between the {splitName} files")

        # Attach The Kaggle Style Columns
        labelFrame = pd.DataFrame({
            "subject": subjects,
            "Activity": [activityLabels[label] for label in labels],
            "Data": self.splitNames[splitName],
        })
        return pd.concat([windowFrame, labelFrame], axis=1)

    def loadCombinedFrame(self):
        """
        This method loads the UCI HAR windows and returns one combined frame

        Arguments
        =========
        None

        Output
        ======
        Tuple of the combined dataframe of train and test windows and the feature name mapping
        """
        # Read The Shared Names And Labels
        featureNames, featureMapping = self.loadFeatureNames()
        activityLabels = self.loadActivityLabels()

        # Concatenate Train Then Test
        splitFrames = [self.loadSplitFrame(splitName, featureNames, activityLabels) for splitName in self.splitNames]
        combinedFrame = pd.concat(splitFrames, ignore_index=True)
        return combinedFrame, featureMapping

    def loadInertialSignal(self, signalName):
        """
        This method loads one raw inertial signal for train then test, in the combined frame row order

        Arguments
        =========
        signalName : Signal file stem, for example body_acc_x

        Output
        ======
        Array with one row per window and one column per raw sample
        """
        # Read Both Splits In The Combined Order
        signalParts = []
        for splitName in self.splitNames:
            signalPath = self.datasetDirectory / splitName / "Inertial Signals" / f"{signalName}_{splitName}.txt"
            signalParts.append(pd.read_csv(signalPath, sep=r"\s+", header=None, dtype=np.float64).to_numpy())
        signalArray = np.vstack(signalParts)

        # Check The Window Length
        if signalArray.shape[1] != self.windowSamples:
            raise ValueError(f"{signalName} has {signalArray.shape[1]} samples per window, expected {self.windowSamples}")
        return signalArray
