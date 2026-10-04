# This file is responsible for keeping the same number of features with four selection methods and scoring each subset
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

from common.csvWriters import writeCsv


# %%
# Selection Sections
class selectionSections:
    def selectFeatures(self):
        """
        This method keeps the same number of features with SelectKBest, RFE, PCA and tree importance and scores one logistic regression on each subset ( chapter Data Feature Selection )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per selection method
        """
        # Score Every Feature With Chi Squared On The Unit Range Features, Which Chi Squared Needs To Be Non Negative
        unitScaler = MinMaxScaler().fit(self.trainFeatures)
        bestSelector = SelectKBest(chi2, k=self.selectedCount).fit(unitScaler.transform(self.trainFeatures), self.trainLabels)
        bestColumns = np.flatnonzero(bestSelector.get_support())
        bestColumns = bestColumns[np.argsort(-bestSelector.scores_[bestColumns], kind="stable")]

        # Eliminate Features Recursively With The Logistic Regression
        eliminator = RFE(self.logisticModel(), n_features_to_select=self.selectedCount, step=self.eliminationStep).fit(self.trainStandard, self.trainLabels)
        eliminationColumns = np.flatnonzero(eliminator.support_)
        eliminationWeights = np.sum(eliminator.estimator_.coef_ ** 2, axis=0)
        eliminationOrder = np.argsort(-eliminationWeights, kind="stable")
        eliminationColumns = eliminationColumns[eliminationOrder]
        eliminationWeights = eliminationWeights[eliminationOrder]

        # Project Onto The Leading Principal Components
        componentModel = PCA(n_components=self.selectedCount, random_state=self.randomSeed).fit(self.trainStandard)

        # Rank The Features By Extra Trees Impurity Importance
        extraTrees = ExtraTreesClassifier(n_estimators=self.forestTrees, random_state=self.randomSeed, n_jobs=self.nJobs).fit(self.trainFeatures, self.trainLabels)
        treeColumns = np.argsort(-extraTrees.feature_importances_, kind="stable")[:self.selectedCount]

        # Write What Every Method Kept
        keptRows = []
        for rankNumber, columnIndex in enumerate(bestColumns, start=1):
            keptRows.append({"method": "SelectKBest", "rank": rankNumber, "kept": self.featureNames[columnIndex], "score": bestSelector.scores_[columnIndex]})
        for rankNumber, (columnIndex, columnWeight) in enumerate(zip(eliminationColumns, eliminationWeights), start=1):
            keptRows.append({"method": "RFE", "rank": rankNumber, "kept": self.featureNames[columnIndex], "score": columnWeight})
        for rankNumber, varianceShare in enumerate(componentModel.explained_variance_ratio_, start=1):
            keptRows.append({"method": "PCA", "rank": rankNumber, "kept": f"component {rankNumber}", "score": varianceShare})
        for rankNumber, columnIndex in enumerate(treeColumns, start=1):
            keptRows.append({"method": "tree importance", "rank": rankNumber, "kept": self.featureNames[columnIndex], "score": extraTrees.feature_importances_[columnIndex]})
        writeCsv(pd.DataFrame(keptRows), self.resultsDirectory / "selectedFeatures.csv")

        # Score The Logistic Regression On Every Subset
        subsetMatrices = {
            "SelectKBest": ("chi2 on unit range features", self.trainStandard[:, bestColumns], self.testStandard[:, bestColumns]),
            "RFE": (f"logistic regression, step {self.eliminationStep:g}", self.trainStandard[:, eliminationColumns], self.testStandard[:, eliminationColumns]),
            "PCA": ("standardized features", componentModel.transform(self.trainStandard), componentModel.transform(self.testStandard)),
            "tree importance": (f"extra trees, {self.forestTrees} trees", self.trainStandard[:, treeColumns], self.testStandard[:, treeColumns]),
        }
        selectionRows = []
        for methodName, (methodSetting, trainMatrix, testMatrix) in subsetMatrices.items():
            subsetScores = self.fitAndScore(self.logisticModel(), trainMatrix, testMatrix)
            selectionRows.append({
                "method": methodName, "setting": methodSetting, "kept": trainMatrix.shape[1],
                "testAccuracy": subsetScores["testAccuracy"], "testMacroF1": subsetScores["testMacroF1"],
            })
            print(f"[ FOUNDATIONS : SELECTION {methodName} MACRO F1 ] : {subsetScores['testMacroF1']:.4f}")
        selectionTable = pd.DataFrame(selectionRows)
        writeCsv(selectionTable, self.resultsDirectory / "featureSelection.csv")
        return selectionTable
