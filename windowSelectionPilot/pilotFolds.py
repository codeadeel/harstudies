# This file is responsible for the pilot steps that train the forest once per participant grouped fold and set the trigger thresholds
# %%
# Importing Libraries
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from common.csvWriters import writeCsv


# %%
# Pilot Folds
class pilotFolds:
    def fitFold(self, foldNumber, trainRows, testRows, windowLabels):
        """
        This method trains the forest on the training windows of one fold and predicts every test window

        Arguments
        =========
        foldNumber : Number of the fold, starting at 1
        trainRows : Rows of the training windows
        testRows : Rows of the test windows
        windowLabels : Label code of every window

        Output
        ======
        Wall clock seconds the forest needed to fit
        """
        # Train The Forest On The Training Windows
        forestModel = RandomForestClassifier(n_estimators=self.forestTrees, random_state=self.randomSeed, n_jobs=self.nJobs)
        fitStart = time.perf_counter()
        forestModel.fit(self.featureMatrix[trainRows], windowLabels[trainRows])
        fitSeconds = time.perf_counter() - fitStart

        # Predict Every Test Window
        cpuStart = time.process_time()
        self.fullPredictions[testRows] = forestModel.predict(self.featureMatrix[testRows])
        self.forestPredictionSeconds += time.process_time() - cpuStart
        self.foldNumbers[testRows] = foldNumber
        return fitSeconds

    def setTriggerThresholds(self, foldNumber, trainRows):
        """
        This method sets the trigger threshold of every budget below 100% as the matching upper quantile of the training changes

        Arguments
        =========
        foldNumber : Number of the fold, starting at 1
        trainRows : Rows of the training windows

        Output
        ======
        List of threshold rows with the fold, the budget and the threshold
        """
        # Set The Trigger Threshold Of Every Budget Below 100% As The Matching Upper Quantile Of The Training Changes
        trainChanges = self.scoreMatrix[trainRows, self.changeColumn]
        trainChanges = trainChanges[~np.isnan(trainChanges)]
        self.triggerThresholds[foldNumber] = {
            budgetPercent: float(np.quantile(trainChanges, 1 - budgetPercent / 100)) if budgetPercent < 100 else -np.inf for budgetPercent in self.budgetPercents
        }
        return [
            {"fold": foldNumber, "budgetPercent": budgetPercent, "threshold": thresholdValue}
            for budgetPercent, thresholdValue in self.triggerThresholds[foldNumber].items() if budgetPercent < 100
        ]

    def summariseFold(self, foldNumber, trainRows, testRows, fitSeconds, windowLabels, participants):
        """
        This method describes one fold with its test participants, its window counts, its fit time and its macro F1

        Arguments
        =========
        foldNumber : Number of the fold, starting at 1
        trainRows : Rows of the training windows
        testRows : Rows of the test windows
        fitSeconds : Wall clock seconds the forest needed to fit
        windowLabels : Label code of every window
        participants : Participant number of every window

        Output
        ======
        Dictionary holding one row of the fold table
        """
        # Describe The Fold
        return {
            "fold": foldNumber,
            "testParticipants": " ".join(str(participantNumber) for participantNumber in sorted(np.unique(participants[testRows]))),
            "trainWindows": len(trainRows),
            "testWindows": len(testRows),
            "forestFitSeconds": fitSeconds,
            "testMacroF1": f1_score(windowLabels[testRows], self.fullPredictions[testRows], average="macro", zero_division=0),
        }

    def runFolds(self):
        """
        This method trains the forest once per participant grouped fold, predicts every test window and sets the trigger thresholds from the training windows

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per fold
        """
        # Prepare The Out Of Fold Arrays
        windowLabels = self.windowTable["label"].to_numpy()
        participants = self.windowTable["participant"].to_numpy()
        self.fullPredictions = np.zeros(len(windowLabels), dtype=windowLabels.dtype)
        self.foldNumbers = np.zeros(len(windowLabels), dtype=np.int64)

        # Train Once Per Fold, Predict Every Test Window And Set The Trigger Thresholds
        foldRows = []
        thresholdRows = []
        for foldNumber, (trainRows, testRows) in enumerate(GroupKFold(n_splits=self.foldCount).split(self.featureMatrix, windowLabels, participants), start=1):
            fitSeconds = self.fitFold(foldNumber, trainRows, testRows, windowLabels)
            thresholdRows += self.setTriggerThresholds(foldNumber, trainRows)
            foldRows.append(self.summariseFold(foldNumber, trainRows, testRows, fitSeconds, windowLabels, participants))
            print(f"[ PILOT : FOLD {foldNumber} MACRO F1 ] : {foldRows[-1]['testMacroF1']:.4f}")

        # Write The Fold Table And The Trigger Thresholds
        foldTable = pd.DataFrame(foldRows)
        writeCsv(foldTable, self.resultsDirectory / "foldSummary.csv")
        writeCsv(pd.DataFrame(thresholdRows), self.resultsDirectory / "triggerThresholds.csv")
        return foldTable
