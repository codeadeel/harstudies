# This file is responsible for loading the UCI HAR frame, the raw accelerations and the recording bouts for the smartphone exploration
# %%
# Importing Libraries
import numpy as np

from common.csvWriters import writeCsv


# %%
# Loading Sections
class loadingSections:
    def loadData(self):
        """
        This method loads the combined UCI HAR frame, the raw body and total acceleration and the recording bouts

        Arguments
        =========
        None

        Output
        ======
        Combined dataframe of train and test windows
        """
        # Load The Combined Frame And Write The Name Mapping
        featureMapping = self.loadFeatureFrame()

        # Load The Raw Body And Total Acceleration In The Same Row Order
        self.loadRawAcceleration()

        # Number The Recording Bouts From The Repeated Half Windows
        continuesNext = self.numberBouts()

        # Record The Dataset Checks
        self.recordDataChecks(featureMapping, continuesNext)
        print(f"[ SMARTPHONE : WINDOWS AND FEATURES ] : {len(self.featureFrame)} x {len(self.featureNames)}")
        return self.featureFrame

    def loadFeatureFrame(self):
        """
        This method loads the combined frame and its feature names and writes the feature name mapping

        Arguments
        =========
        None

        Output
        ======
        Dataframe mapping every original feature name to its unique name
        """
        # Load The Combined Frame And Write The Name Mapping
        self.featureFrame, featureMapping = self.loader.loadCombinedFrame()
        self.featureNames = featureMapping["uniqueName"].tolist()
        self.activityOrder = list(self.loader.loadActivityLabels().values())
        writeCsv(featureMapping, self.resultsDirectory / "featureNameMapping.csv")
        return featureMapping

    def loadRawAcceleration(self):
        """
        This method loads the raw body and total acceleration of every axis in the combined frame row order

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Load The Raw Body And Total Acceleration In The Same Row Order
        self.bodyAcceleration = [self.loader.loadInertialSignal(f"body_acc_{axisName}") for axisName in "xyz"]
        self.totalAcceleration = [self.loader.loadInertialSignal(f"total_acc_{axisName}") for axisName in "xyz"]

    def numberBouts(self):
        """
        This method finds the neighbouring windows that continue each other and numbers the recording bouts

        Arguments
        =========
        None

        Output
        ======
        Boolean array whose entry i is true when row i + 1 continues row i within one recording
        """
        # Number The Recording Bouts From The Repeated Half Windows
        subjects = self.featureFrame["subject"].to_numpy()
        splits = self.featureFrame["Data"].to_numpy()
        continuesNext = self.loader.findContinuedWindows(
            self.bodyAcceleration, subjects, self.featureFrame["Activity"].to_numpy(), splits
        )
        self.boutNumbers = self.loader.labelBouts(continuesNext)
        return continuesNext

    def recordDataChecks(self, featureMapping, continuesNext):
        """
        This method records the window, subject, feature, name and bout counts for runParameters.csv

        Arguments
        =========
        featureMapping : Dataframe mapping every original feature name to its unique name
        continuesNext : Boolean array from numberBouts

        Output
        ======
        None
        """
        # Read The Subjects And Splits Of Every Window
        subjects = self.featureFrame["subject"].to_numpy()
        splits = self.featureFrame["Data"].to_numpy()

        # Record The Dataset Checks
        trainSubjects = set(subjects[splits == "Train"].tolist())
        testSubjects = set(subjects[splits == "Test"].tolist())
        self.recordParameter("data", "windows", len(self.featureFrame))
        self.recordParameter("data", "trainWindows", int(np.sum(splits == "Train")))
        self.recordParameter("data", "testWindows", int(np.sum(splits == "Test")))
        self.recordParameter("data", "features", len(self.featureNames))
        self.recordParameter("data", "subjects", len(trainSubjects | testSubjects))
        self.recordParameter("data", "trainSubjects", len(trainSubjects))
        self.recordParameter("data", "testSubjects", len(testSubjects))
        self.recordParameter("data", "sharedSubjects", len(trainSubjects & testSubjects))
        self.recordParameter("data", "repeatedFeatureNames", featureMapping.loc[featureMapping["nameCount"] > 1, "originalName"].nunique())
        self.recordParameter("data", "renamedFeatures", int(featureMapping["renamed"].sum()))
        renamedNames = featureMapping.loc[featureMapping["renamed"], "uniqueName"]
        repeatedNames = featureMapping.loc[featureMapping["nameCount"] > 1, "originalName"]
        self.recordParameter("data", "renamedSuffixes", ", ".join(sorted({"." + uniqueName.rsplit(".", 1)[1] for uniqueName in renamedNames})))
        self.recordParameter("data", "repeatedNameFunctions", ", ".join(sorted({originalName.split("-")[1] for originalName in repeatedNames})))
        self.recordParameter("data", "samplingRateHz", self.loader.samplingRate)
        self.recordParameter("data", "windowSamples", self.loader.windowSamples)
        self.recordParameter("data", "windowSeconds", self.loader.windowSeconds)
        self.recordParameter("data", "windowStepSeconds", self.loader.windowStepSeconds)
        self.recordParameter("data", "neighbourPairs", len(continuesNext))
        self.recordParameter("data", "continuedPairs", int(continuesNext.sum()))
        self.recordParameter("data", "recordingBouts", int(self.boutNumbers.max() + 1))
