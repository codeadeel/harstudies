# This file is responsible for scoring every selection method at every budget and writing the budget, survival and recall tables
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common.csvWriters import writeCsv


# %%
# Pilot Scoring
class pilotScoring:
    def scoreSelection(self, methodName, budgetPercent, seedOffset, windowLabels, labelCodes, activityMasks):
        """
        This method scores one method at one budget and one random seed on the pooled out of fold predictions

        Arguments
        =========
        methodName : Selection method
        budgetPercent : Budget in percent
        seedOffset : Offset added to the seed for the random baseline, None for the other methods
        windowLabels : Label code of every window
        labelCodes : Array of all label codes
        activityMasks : List with one boolean window mask per activity

        Output
        ======
        Tuple of the run row, the list of survival rows and the list of recall rows
        """
        # Select The Windows And Carry The Predictions Forward
        keptMask, finalPredictions = self.selectAll(methodName, budgetPercent, seedOffset)

        # Check That The Full Budget Reproduces The Recognizer
        if budgetPercent == 100 and not (keptMask.all() and np.array_equal(finalPredictions, self.fullPredictions)):
            raise RuntimeError(f"{methodName} at 100% differs from the recognizer")

        # Keep The Pooled Scores, The Survival And The Recall Of Every Activity
        seedValue = np.nan if seedOffset is None else self.randomSeed + seedOffset
        runRow = {
            "method": methodName, "budgetPercent": budgetPercent, "seed": seedValue, "realizedBudget": keptMask.mean(),
            "macroF1": f1_score(windowLabels, finalPredictions, average="macro", labels=labelCodes, zero_division=0),
            "accuracy": accuracy_score(windowLabels, finalPredictions),
        }
        survivalRows, recallRows = [], []
        for labelCode, activityMask in zip(labelCodes, activityMasks):
            survivalRows.append({
                "method": methodName, "budgetPercent": budgetPercent, "seed": seedValue, "activity": self.builder.labelNames[labelCode],
                "windows": int(activityMask.sum()), "keptShare": keptMask[activityMask].mean(),
            })
            if budgetPercent in (self.reportPercent, 100):
                recallRows.append({
                    "method": methodName, "budgetPercent": budgetPercent, "seed": seedValue, "activity": self.builder.labelNames[labelCode],
                    "windows": int(activityMask.sum()), "recall": np.mean(finalPredictions[activityMask] == labelCode),
                })
        return runRow, survivalRows, recallRows

    def scoreMethods(self):
        """
        This method scores every method at every budget and random seed

        Arguments
        =========
        None

        Output
        ======
        Tuple of the lists of run rows, survival rows and recall rows
        """
        # Prepare The Activity Masks
        windowLabels = self.windowTable["label"].to_numpy()
        labelCodes = np.arange(len(self.builder.labelNames))
        activityMasks = [windowLabels == labelCode for labelCode in labelCodes]

        # Score Every Method, Budget And Random Seed
        runRows, survivalRows, recallRows = [], [], []
        for methodName in self.methodNames:
            seedOffsets = list(range(self.randomSeedCount)) if methodName == "random" else [None]
            for budgetPercent in self.budgetPercents:
                for seedOffset in seedOffsets:
                    runRow, runSurvivalRows, runRecallRows = self.scoreSelection(methodName, budgetPercent, seedOffset, windowLabels, labelCodes, activityMasks)
                    runRows.append(runRow)
                    survivalRows += runSurvivalRows
                    recallRows += runRecallRows
            print(f"[ PILOT : METHOD SCORED ] : {methodName}")
        return runRows, survivalRows, recallRows

    def writeScoreTables(self, runRows, survivalRows, recallRows):
        """
        This method writes the runs and the tables that average the random seeds

        Arguments
        =========
        runRows : List of run rows
        survivalRows : List of survival rows
        recallRows : List of recall rows

        Output
        ======
        Dataframe of macro F1 against budget per method
        """
        # Average The Random Seeds
        runTable = pd.DataFrame(runRows)
        writeCsv(runTable, self.resultsDirectory / "budgetRuns.csv")
        curveTable = runTable.groupby(["method", "budgetPercent"], sort=False).agg(
            realizedBudget=("realizedBudget", "mean"), macroF1=("macroF1", "mean"), macroF1Std=("macroF1", "std"), accuracy=("accuracy", "mean"), runs=("macroF1", "size"),
        ).reset_index()
        writeCsv(curveTable, self.resultsDirectory / "budgetCurves.csv")
        survivalTable = pd.DataFrame(survivalRows).groupby(["method", "budgetPercent", "activity"], sort=False).agg(
            windows=("windows", "first"), keptShare=("keptShare", "mean"),
        ).reset_index()
        writeCsv(survivalTable, self.resultsDirectory / "survivalShares.csv")
        recallTable = pd.DataFrame(recallRows).groupby(["method", "budgetPercent", "activity"], sort=False).agg(
            windows=("windows", "first"), recall=("recall", "mean"),
        ).reset_index()
        writeCsv(recallTable, self.resultsDirectory / "activityRecall.csv")
        return curveTable

    def evaluateSelections(self):
        """
        This method scores every method at every budget on the pooled out of fold predictions and measures the per activity recall and survival

        Arguments
        =========
        None

        Output
        ======
        Dataframe of macro F1 against budget per method
        """
        # Score Every Method, Budget And Random Seed
        runRows, survivalRows, recallRows = self.scoreMethods()

        # Average The Random Seeds And Write The Tables
        return self.writeScoreTables(runRows, survivalRows, recallRows)
