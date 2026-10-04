# This file is responsible for scoring every fold with the accuracy and the class averaged metrics and pooling the confusion counts
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Fold Scoring Steps
class foldScoringSteps:
    def scoreFold(self, trueLabels, predictedLabels):
        """
        This method computes the accuracy and the class averaged precision, recall, specificity and F-measures of one fold

        Arguments
        =========
        trueLabels : True label of every test row
        predictedLabels : Predicted label of every test row

        Output
        ======
        Tuple of the metric dictionary and the confusion counts ( true classes as rows )
        """
        # Count The Confusion And The Per Class Outcomes
        classLabels = np.asarray(self.loader.classLabels)
        confusionCounts = np.zeros((len(classLabels), len(classLabels)), dtype=np.int64)
        np.add.at(confusionCounts, (np.searchsorted(classLabels, trueLabels), np.searchsorted(classLabels, predictedLabels)), 1)
        truePositives = np.diag(confusionCounts).astype(np.float64)
        falsePositives = confusionCounts.sum(axis=0) - truePositives
        falseNegatives = confusionCounts.sum(axis=1) - truePositives
        trueNegatives = confusionCounts.sum() - truePositives - falsePositives - falseNegatives

        # Average Over Classes, A Class Never Predicted Having Precision Zero
        classPrecision = np.divide(truePositives, truePositives + falsePositives, out=np.zeros_like(truePositives), where=truePositives + falsePositives > 0)
        classRecall = np.divide(truePositives, truePositives + falseNegatives, out=np.zeros_like(truePositives), where=truePositives + falseNegatives > 0)
        classSpecificity = np.divide(trueNegatives, trueNegatives + falsePositives, out=np.zeros_like(truePositives), where=trueNegatives + falsePositives > 0)
        classF1 = np.divide(2 * classPrecision * classRecall, classPrecision + classRecall, out=np.zeros_like(truePositives), where=classPrecision + classRecall > 0)
        macroPrecision, macroRecall = classPrecision.mean(), classRecall.mean()
        return {
            "accuracy": truePositives.sum() / confusionCounts.sum(), "macroPrecision": macroPrecision, "macroRecall": macroRecall,
            "macroSpecificity": classSpecificity.mean(), "macroF1": classF1.mean(),
            "attalF": (1 + self.fBeta ** 2) * macroPrecision * macroRecall / (self.fBeta ** 2 * macroPrecision + macroRecall) if macroPrecision + macroRecall > 0 else 0.0,
        }, confusionCounts

    def scoreRecords(self, variants, foldRecords):
        """
        This method scores every fold record and pools the confusion counts per variant, protocol and classifier

        Arguments
        =========
        variants : Output of prepareData
        foldRecords : Fold records of both evaluations

        Output
        ======
        Tuple of the fold metric table and the pooled confusion counts
        """
        # Score Each Fold Against The Labels Of Its Test Rows
        metricRows, pooledConfusions = self.scoreEveryFold(variants, foldRecords)
        metricTable = self.setMetricTypes(self.sortRecords(pd.DataFrame(metricRows)))
        writeCsv(metricTable.drop(columns=["keptFeatureNames"]), self.resultsDirectory / "foldMetrics.csv")

        # Keep The Kept Features Of Every Fold For Inspection
        selectionTable = metricTable[metricTable["keptFeatureNames"] != ""].drop_duplicates(["variant", "protocol", "fold"])[
            ["variant", "protocol", "fold", "keptFeatures", "keptFeatureNames"]
        ]
        writeCsv(selectionTable, self.resultsDirectory / "featureSelectionFolds.csv")
        return metricTable, pooledConfusions

    def scoreEveryFold(self, variants, foldRecords):
        """
        This method scores every fold record against the labels of its test rows and pools the confusion counts

        Arguments
        =========
        variants : Output of prepareData
        foldRecords : Fold records of both evaluations

        Output
        ======
        Tuple of the list of metric rows and the pooled confusion counts per variant, protocol and classifier
        """
        # Score Each Fold Against The Labels Of Its Test Rows
        metricRows, pooledConfusions = [], {}
        for foldRecord in foldRecords:
            trueLabels = variants[foldRecord["variant"]]["rows"]["label"].to_numpy()[foldRecord["testPositions"]]
            foldMetrics, confusionCounts = self.scoreFold(trueLabels, foldRecord["predictions"])
            poolKey = (foldRecord["variant"], foldRecord["protocol"], foldRecord["classifier"])
            pooledConfusions[poolKey] = pooledConfusions.get(poolKey, 0) + confusionCounts
            metricRows.append({
                columnName: columnValue for columnName, columnValue in foldRecord.items() if columnName not in ("testPositions", "predictions")
            } | foldMetrics)
        return metricRows, pooledConfusions

    def setMetricTypes(self, metricTable):
        """
        This method gives the integer and boolean columns of the fold metric table their nullable types

        Arguments
        =========
        metricTable : Fold metric table in its fixed order

        Output
        ======
        The same table with typed columns
        """
        # Use Nullable Integers And Booleans Where A Column Is Empty For Some Classifiers
        for columnName in ["keptFeatures", "neighbours", "trees", "log2Cost", "log2Gamma", "emIterations"]:
            if columnName in metricTable.columns:
                metricTable[columnName] = metricTable[columnName].astype("Int64")
        for columnName in ["converged", "emConverged"]:
            if columnName in metricTable.columns:
                metricTable[columnName] = metricTable[columnName].astype("boolean")
        return metricTable
