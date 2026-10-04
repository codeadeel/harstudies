# This file is responsible for creating the shared logistic regression and for fitting and scoring one classifier
# %%
# Importing Libraries
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# %%
# Model Helpers
class modelHelpers:
    def logisticModel(self):
        """
        This method creates the logistic regression used by several chapters

        Arguments
        =========
        None

        Output
        ======
        Unfitted logistic regression
        """
        # Create The Model
        return LogisticRegression(max_iter=self.logisticIterations)

    def fitAndScore(self, model, trainMatrix, testMatrix):
        """
        This method fits a classifier on the training windows, times the fit and scores it on the official test windows

        Arguments
        =========
        model : Unfitted scikit-learn classifier
        trainMatrix : Training feature matrix
        testMatrix : Test feature matrix with the same columns

        Output
        ======
        Dictionary with the fit seconds, the test accuracy and the test macro F1
        """
        # Fit The Model And Time The Fit
        fitStart = time.perf_counter()
        model.fit(trainMatrix, self.trainLabels)
        fitSeconds = time.perf_counter() - fitStart

        # Score The Official Test Split
        testPredictions = model.predict(testMatrix)
        return {
            "fitSeconds": fitSeconds,
            "testAccuracy": accuracy_score(self.testLabels, testPredictions),
            "testMacroF1": f1_score(self.testLabels, testPredictions, average="macro"),
        }
