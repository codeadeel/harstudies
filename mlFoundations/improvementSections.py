# This file is responsible for comparing the ensembles and tuning the support vector machine with participant grouped folds
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.csvWriters import writeCsv


# %%
# Improvement Sections
class improvementSections:
    def compareEnsembles(self):
        """
        This method fits bagging, a random forest, AdaBoost and a voting ensemble and scores them on the test windows ( chapter Improving Performance of ML Models )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per ensemble
        """
        # Describe The Four Ensembles
        ensembles = {
            "bagging": (
                BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=self.ensembleTrees, random_state=self.randomSeed, n_jobs=self.nJobs),
                f"{self.ensembleTrees} decision trees",
            ),
            "randomForest": (
                RandomForestClassifier(n_estimators=self.ensembleTrees, random_state=self.randomSeed, n_jobs=self.nJobs),
                f"{self.ensembleTrees} trees",
            ),
            "adaBoost": (
                AdaBoostClassifier(n_estimators=self.boostingRounds, random_state=self.randomSeed),
                f"{self.boostingRounds} decision stumps",
            ),
            "voting": (
                VotingClassifier([("logistic", self.logisticModel()), ("tree", DecisionTreeClassifier(random_state=self.randomSeed)), ("svm", SVC())], voting="hard", n_jobs=self.nJobs),
                "hard vote of logistic regression, decision tree and RBF SVM",
            ),
        }

        # Fit And Score Every Ensemble
        ensembleRows = []
        for ensembleName, (ensembleModel, ensembleSetting) in ensembles.items():
            ensembleScores = self.fitAndScore(ensembleModel, self.trainStandard, self.testStandard)
            ensembleRows.append({"ensemble": ensembleName, "setting": ensembleSetting, **ensembleScores})
            print(f"[ FOUNDATIONS : ENSEMBLE {ensembleName} ACCURACY ] : {ensembleScores['testAccuracy']:.4f}")
        ensembleTable = pd.DataFrame(ensembleRows)
        writeCsv(ensembleTable, self.resultsDirectory / "ensembleComparison.csv")
        return ensembleTable

    def tuneWithGroups(self):
        """
        This method tunes C and gamma of the RBF SVM with GridSearchCV over participant grouped folds of the training windows and scores the refitted best model on the test windows ( chapter Improving Performance of ML Model, continued )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per grid point
        """
        # Search The Grid With Folds That Keep Each Participant Together
        searchModel = GridSearchCV(
            Pipeline([("standardize", StandardScaler()), ("svm", SVC())]),
            {"svm__C": self.gridCosts, "svm__gamma": self.gridGammas},
            scoring="f1_macro", cv=GroupKFold(n_splits=self.groupFolds), n_jobs=self.nJobs, refit=True,
        )
        searchModel.fit(self.trainFeatures, self.trainLabels, groups=self.trainSubjects)

        # Write Every Grid Point
        searchResults = searchModel.cv_results_
        gridTable = pd.DataFrame({
            "C": searchResults["param_svm__C"].astype(float),
            "gamma": searchResults["param_svm__gamma"].astype(float),
            "groupFoldMacroF1Mean": searchResults["mean_test_score"],
            "groupFoldMacroF1Std": searchResults["std_test_score"],
            "rank": searchResults["rank_test_score"],
        })
        writeCsv(gridTable, self.resultsDirectory / "gridSearch.csv")

        # Score The Refitted Best Model On The Test Windows
        testPredictions = searchModel.predict(self.testFeatures)
        bestTable = pd.DataFrame([{
            "C": float(searchModel.best_params_["svm__C"]), "gamma": float(searchModel.best_params_["svm__gamma"]),
            "groupFoldMacroF1Mean": searchModel.best_score_,
            "testAccuracy": accuracy_score(self.testLabels, testPredictions),
            "testMacroF1": f1_score(self.testLabels, testPredictions, average="macro"),
        }])
        writeCsv(bestTable, self.resultsDirectory / "gridSearchBest.csv")
        self.recordParameter("gridSearch", "groupFolds", self.groupFolds)
        self.recordParameter("gridSearch", "trainingParticipants", len(np.unique(self.trainSubjects)))
        print(f"[ FOUNDATIONS : GRID SEARCH BEST ] : C {searchModel.best_params_['svm__C']} gamma {searchModel.best_params_['svm__gamma']}")
        return gridTable
