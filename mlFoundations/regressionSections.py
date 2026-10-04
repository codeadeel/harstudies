# This file is responsible for fitting two regressors on the diabetes data and scoring them on a held out share
# %%
# Importing Libraries
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common.csvWriters import writeCsv


# %%
# Regression Sections
class regressionSections:
    def runRegression(self):
        """
        This method fits a linear regression and a random forest regressor on scikit-learn's diabetes data and scores them on a held out share ( chapter Regression Algorithms )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per regressor
        """
        # Split The Diabetes Data
        diabetesData = load_diabetes()
        trainInputs, testInputs, trainTargets, testTargets = train_test_split(
            diabetesData.data, diabetesData.target, test_size=self.regressionTestShare, random_state=self.randomSeed,
        )
        self.recordParameter("regression", "rows", diabetesData.data.shape[0])
        self.recordParameter("regression", "inputs", diabetesData.data.shape[1])
        self.recordParameter("regression", "testShare", self.regressionTestShare)
        self.recordParameter("regression", "testRows", len(testTargets))

        # Fit And Score Both Regressors
        regressionRows = []
        regressors = {
            "linearRegression": (LinearRegression(), "ordinary least squares"),
            "randomForest": (RandomForestRegressor(n_estimators=self.forestTrees, random_state=self.randomSeed, n_jobs=self.nJobs), f"{self.forestTrees} trees"),
        }
        for regressorName, (regressorModel, regressorSetting) in regressors.items():
            testPredictions = regressorModel.fit(trainInputs, trainTargets).predict(testInputs)
            regressionRows.append({
                "regressor": regressorName, "setting": regressorSetting,
                "meanAbsoluteError": mean_absolute_error(testTargets, testPredictions),
                "meanSquaredError": mean_squared_error(testTargets, testPredictions),
                "r2": r2_score(testTargets, testPredictions),
            })
        regressionTable = pd.DataFrame(regressionRows)
        writeCsv(regressionTable, self.resultsDirectory / "regressionMetrics.csv")
        return regressionTable
