# This file is responsible for fitting the six classifiers of the tutorial and scoring them on the test windows
# %%
# Importing Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.csvWriters import writeCsv


# %%
# Classifier Sections
class classifierSections:
    def compareClassifiers(self):
        """
        This method fits the six classifiers of the tutorial on the standardized training windows and scores them on the test windows ( chapters Classification Algorithms and KNN Algorithm )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per classifier
        """
        # Describe The Six Classifiers
        classifierSettings = {
            "logisticRegression": (self.logisticModel(), f"lbfgs, max_iter {self.logisticIterations}"),
            "supportVectorMachine": (SVC(), "RBF kernel, C 1, gamma scale"),
            "decisionTree": (DecisionTreeClassifier(random_state=self.randomSeed), "default"),
            "naiveBayes": (GaussianNB(), "Gaussian"),
            "randomForest": (RandomForestClassifier(n_estimators=self.forestTrees, random_state=self.randomSeed, n_jobs=self.nJobs), f"{self.forestTrees} trees"),
            "kNearestNeighbours": (KNeighborsClassifier(n_neighbors=self.neighbourCount, n_jobs=self.nJobs), f"K {self.neighbourCount}"),
        }

        # Fit And Score Every Classifier
        comparisonRows = []
        for classifierName, (classifierModel, classifierSetting) in classifierSettings.items():
            classifierScores = self.fitAndScore(classifierModel, self.trainStandard, self.testStandard)
            self.fittedClassifiers[classifierName] = classifierModel
            comparisonRows.append({"classifier": classifierName, "setting": classifierSetting, **classifierScores})
            print(f"[ FOUNDATIONS : CLASSIFIER {classifierName} ACCURACY ] : {classifierScores['testAccuracy']:.4f}")
        comparisonTable = pd.DataFrame(comparisonRows)
        writeCsv(comparisonTable, self.resultsDirectory / "classifierComparison.csv")
        return comparisonTable
