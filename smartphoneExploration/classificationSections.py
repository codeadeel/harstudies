# This file is responsible for classifying the activities with LightGBM on the dataset's own train and test split for the smartphone exploration
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from common.csvWriters import writeCsv
from common.plotDefaults import saveFigure


# %%
# Classification Sections
class classificationSections:
    def classifyActivities(self):
        """
        This method trains LightGBM on the dataset's own subject disjoint train split and tests it on the test split ( section 4.2 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the activity classification metrics
        """
        # Record The Settings Added To The LightGBM Defaults
        self.recordParameter("lightgbm", "addedSettings", ", ".join(f"{settingName}={settingValue}" for settingName, settingValue in self.classifierSettings.items()))

        # Split By The Dataset's Own Train And Test Columns
        trainMask = (self.featureFrame["Data"] == "Train").to_numpy()
        featureMatrix = self.featureFrame[self.featureNames].to_numpy()
        activities = self.featureFrame["Activity"].to_numpy()
        subjects = self.featureFrame["subject"].to_numpy()

        # Train And Test The Classifier
        activityModel = self.buildClassifier().fit(featureMatrix[trainMask], activities[trainMask])
        predictedActivities = activityModel.predict(featureMatrix[~trainMask])
        trueActivities = activities[~trainMask]

        # Write The Headline Metrics
        metricTable = self.writeClassificationMetrics(trainMask, subjects, trueActivities, predictedActivities)

        # Write The Confusion Matrix With True Activities As Rows
        confusionCounts = self.writeConfusionMatrix(trueActivities, predictedActivities)

        # Write The Per Activity Scores
        self.writeClassReport(trueActivities, predictedActivities)

        # Plot The Confusion Matrix
        self.plotConfusionMatrix(confusionCounts)
        print(f"[ SMARTPHONE : ACTIVITY ACCURACY ] : {metricTable.loc[0, 'accuracy']:.4f}")
        return metricTable

    def writeClassificationMetrics(self, trainMask, subjects, trueActivities, predictedActivities):
        """
        This method writes the accuracy, macro F1 and the split sizes of the activity classifier

        Arguments
        =========
        trainMask : Boolean array that marks the training windows
        subjects : Subject of every window
        trueActivities : True activity of every test window
        predictedActivities : Predicted activity of every test window

        Output
        ======
        Dataframe of the activity classification metrics
        """
        metricTable = pd.DataFrame([{
            "accuracy": accuracy_score(trueActivities, predictedActivities),
            "macroF1": f1_score(trueActivities, predictedActivities, average="macro"),
            "trainWindows": int(trainMask.sum()),
            "testWindows": int((~trainMask).sum()),
            "trainSubjects": len(np.unique(subjects[trainMask])),
            "testSubjects": len(np.unique(subjects[~trainMask])),
            "sharedSubjects": len(np.intersect1d(subjects[trainMask], subjects[~trainMask])),
        }])
        writeCsv(metricTable, self.resultsDirectory / "activityClassification.csv")
        return metricTable

    def writeConfusionMatrix(self, trueActivities, predictedActivities):
        """
        This method writes the confusion matrix with true activities as rows

        Arguments
        =========
        trueActivities : True activity of every test window
        predictedActivities : Predicted activity of every test window

        Output
        ======
        Array of the confusion counts in activity order
        """
        confusionCounts = confusion_matrix(trueActivities, predictedActivities, labels=self.activityOrder)
        confusionTable = pd.DataFrame(confusionCounts, index=self.activityOrder, columns=self.activityOrder).rename_axis("trueActivity").reset_index()
        writeCsv(confusionTable, self.resultsDirectory / "activityConfusionMatrix.csv")
        return confusionCounts

    def writeClassReport(self, trueActivities, predictedActivities):
        """
        This method writes the precision, recall and F1 score of every activity

        Arguments
        =========
        trueActivities : True activity of every test window
        predictedActivities : Predicted activity of every test window

        Output
        ======
        None
        """
        precisionScores, recallScores, f1Scores, supportCounts = precision_recall_fscore_support(
            trueActivities, predictedActivities, labels=self.activityOrder, zero_division=0
        )
        classTable = pd.DataFrame({
            "Activity": self.activityOrder, "precision": precisionScores, "recall": recallScores,
            "f1": f1Scores, "testWindows": supportCounts,
        })
        writeCsv(classTable, self.resultsDirectory / "activityClassReport.csv")

    def plotConfusionMatrix(self, confusionCounts):
        """
        This method draws the confusion matrix with its counts

        Arguments
        =========
        confusionCounts : Array of the confusion counts in activity order

        Output
        ======
        None
        """
        # Plot The Confusion Matrix
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.imshow(confusionCounts, cmap="Blues")
        axis.grid(False)
        axis.spines[:].set_visible(True)
        for rowIndex in range(len(self.activityOrder)):
            for columnIndex in range(len(self.activityOrder)):
                cellCount = confusionCounts[rowIndex, columnIndex]
                axis.text(columnIndex, rowIndex, str(cellCount), ha="center", va="center", fontsize=8,
                          color="white" if cellCount > confusionCounts.max() / 2 else "black")
        axis.set_xticks(range(len(self.activityOrder)), self.activityOrder, rotation=30, ha="right")
        axis.set_yticks(range(len(self.activityOrder)), self.activityOrder)
        axis.set_xlabel("Predicted activity")
        axis.set_ylabel("True activity")
        axis.set_title("LightGBM on the dataset's test subjects")
        saveFigure(figure, self.figuresDirectory / "activityConfusionMatrix.png")
