# This file is responsible for writing the confusion matrix, the per class report and the probability scores of the logistic regression
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score

from common.csvWriters import writeCsv
from common.plotDefaults import saveFigure


# %%
# Metrics Sections
class metricsSections:
    def evaluateMetrics(self):
        """
        This method writes the confusion matrix, a per class report with specificity, ROC AUC and log loss of the logistic regression on the test windows ( chapter Performance Metrics )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the per class report
        """
        # Predict The Test Windows With The Fitted Logistic Regression
        logisticModel = self.fittedClassifiers["logisticRegression"]
        testPredictions = logisticModel.predict(self.testStandard)
        testProbabilities = logisticModel.predict_proba(self.testStandard)

        # Write The Confusion Matrix In Activity Order
        countMatrix = confusion_matrix(self.testLabels, testPredictions, labels=self.activityOrder)
        confusionTable = pd.DataFrame(countMatrix, columns=self.activityOrder)
        confusionTable.insert(0, "activity", self.activityOrder)
        writeCsv(confusionTable, self.resultsDirectory / "confusionMatrix.csv")

        # Write The Per Class Report
        reportTable = self.writeClassificationReport(countMatrix, testPredictions)

        # Score The Probabilities
        probabilityTable = pd.DataFrame([
            {"metric": "accuracy", "value": accuracy_score(self.testLabels, testPredictions)},
            {"metric": "ROC AUC, one against the rest, macro average", "value": roc_auc_score(self.testLabels, testProbabilities, multi_class="ovr", average="macro", labels=logisticModel.classes_)},
            {"metric": "log loss", "value": log_loss(self.testLabels, testProbabilities, labels=logisticModel.classes_)},
        ])
        writeCsv(probabilityTable, self.resultsDirectory / "probabilityMetrics.csv")

        # Draw The Confusion Matrix Figure
        self.plotConfusionMatrix(countMatrix)
        return reportTable

    def writeClassificationReport(self, countMatrix, testPredictions):
        """
        This method writes precision, recall, F1 and specificity per class with a macro and a weighted average

        Arguments
        =========
        countMatrix : Confusion matrix in activity order
        testPredictions : Predicted activity of every test window

        Output
        ======
        Dataframe of the per class report
        """
        # Report Precision, Recall, F1 And Specificity Per Class
        precisionValues, recallValues, f1Values, supportValues = precision_recall_fscore_support(self.testLabels, testPredictions, labels=self.activityOrder)
        falsePositives = countMatrix.sum(axis=0) - np.diag(countMatrix)
        trueNegatives = countMatrix.sum() - countMatrix.sum(axis=1) - countMatrix.sum(axis=0) + np.diag(countMatrix)
        specificityValues = trueNegatives / (trueNegatives + falsePositives)
        reportTable = pd.DataFrame({
            "activity": self.activityOrder, "precision": precisionValues, "recall": recallValues, "f1": f1Values,
            "specificity": specificityValues, "support": supportValues,
        })
        averageRows = pd.DataFrame([
            {"activity": "macro average", **{metricName: reportTable[metricName].mean() for metricName in ["precision", "recall", "f1", "specificity"]}, "support": int(supportValues.sum())},
            {"activity": "weighted average", **{metricName: np.average(reportTable[metricName], weights=supportValues) for metricName in ["precision", "recall", "f1", "specificity"]}, "support": int(supportValues.sum())},
        ])
        reportTable = pd.concat([reportTable, averageRows], ignore_index=True)
        writeCsv(reportTable, self.resultsDirectory / "classificationReport.csv")
        return reportTable

    def plotConfusionMatrix(self, countMatrix):
        """
        This method draws the confusion matrix with the count of every cell

        Arguments
        =========
        countMatrix : Confusion matrix in activity order

        Output
        ======
        None
        """
        # Plot The Confusion Matrix
        figure, axis = plt.subplots(figsize=(7, 6))
        matrixImage = axis.imshow(countMatrix, cmap="Blues")
        for rowIndex in range(len(self.activityOrder)):
            for columnIndex in range(len(self.activityOrder)):
                cellCount = countMatrix[rowIndex, columnIndex]
                axis.text(columnIndex, rowIndex, str(cellCount), ha="center", va="center", fontsize=8, color="white" if cellCount > countMatrix.max() / 2 else "black")
        axis.set_xticks(range(len(self.activityOrder)), labels=self.activityOrder, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(self.activityOrder)), labels=self.activityOrder, fontsize=8)
        axis.set_xlabel("Predicted activity")
        axis.set_ylabel("True activity")
        axis.grid(False)
        figure.colorbar(matrixImage, ax=axis, label="Test windows")
        axis.set_title("Logistic regression on the test windows")
        saveFigure(figure, self.figuresDirectory / "confusionMatrix.png")
