# This file is responsible for loading the UCI HAR windows, splitting them and writing the loading overview
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.csvWriters import writeCsv


# %%
# Loading Sections
class loadingSections:
    def loadData(self):
        """
        This method loads the UCI HAR windows the pandas way through the UCI HAR loader of smartphoneExploration and checks NumPy reads the same values ( chapter Data Loading )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the loading overview
        """
        # Load The Windows With pandas Through The smartphoneExploration Loader
        self.featureFrame = self.loader.loadCombinedFrame()[0]
        self.activityOrder = list(self.loader.loadActivityLabels().values())
        self.featureNames = [columnName for columnName in self.featureFrame.columns if columnName not in ("subject", "Activity", "Data")]

        # Split Into The Official Training And Test Windows
        trainMask = (self.featureFrame["Data"] == "Train").to_numpy()
        self.trainFeatures = self.featureFrame.loc[trainMask, self.featureNames].to_numpy()
        self.testFeatures = self.featureFrame.loc[~trainMask, self.featureNames].to_numpy()
        self.trainLabels = self.featureFrame.loc[trainMask, "Activity"].to_numpy()
        self.testLabels = self.featureFrame.loc[~trainMask, "Activity"].to_numpy()
        self.trainSubjects = self.featureFrame.loc[trainMask, "subject"].to_numpy()
        testSubjects = self.featureFrame.loc[~trainMask, "subject"].to_numpy()

        # Standardise With The Training Windows For The Later Chapters
        standardScaler = StandardScaler().fit(self.trainFeatures)
        self.trainStandard = standardScaler.transform(self.trainFeatures)
        self.testStandard = standardScaler.transform(self.testFeatures)

        # Check That NumPy Reads The Same Training Values
        numpyFeatures = np.loadtxt(self.loader.datasetDirectory / "train" / "X_train.txt")
        largestDifference = float(np.max(np.abs(numpyFeatures - self.trainFeatures)))

        # Write The Loading Overview
        featureTypes = self.featureFrame[self.featureNames].dtypes
        overviewTable = pd.DataFrame([
            {"item": "training windows", "value": int(trainMask.sum())},
            {"item": "test windows", "value": int((~trainMask).sum())},
            {"item": "training participants", "value": len(np.unique(self.trainSubjects))},
            {"item": "test participants", "value": len(np.unique(testSubjects))},
            {"item": "participants in both splits", "value": len(set(self.trainSubjects) & set(testSubjects))},
            {"item": "feature columns", "value": len(self.featureNames)},
            {"item": "float64 feature columns", "value": int((featureTypes == np.float64).sum())},
            {"item": "missing feature values", "value": int(self.featureFrame[self.featureNames].isna().sum().sum())},
            {"item": "activities", "value": len(self.activityOrder)},
            {"item": "largest absolute difference between numpy.loadtxt and pandas on X_train.txt", "value": largestDifference},
        ])
        writeCsv(overviewTable, self.resultsDirectory / "dataOverview.csv")
        print(f"[ FOUNDATIONS : WINDOWS AND FEATURES ] : {len(self.featureFrame)} x {len(self.featureNames)}")
        return overviewTable
