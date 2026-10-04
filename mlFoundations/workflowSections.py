# This file is responsible for building one Pipeline and one FeatureUnion and scoring them with participant grouped folds
# %%
# Importing Libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from common.csvWriters import writeCsv


# %%
# Workflow Sections
class workflowSections:
    def buildPipelines(self):
        """
        This method builds one Pipeline and one FeatureUnion, scores them with participant grouped folds inside the training windows and on the test windows ( chapter Automatic Workflows )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per workflow
        """
        # Build The Two Workflows
        workflows = {
            "Pipeline": (
                Pipeline([("standardize", StandardScaler()), ("discriminant", LinearDiscriminantAnalysis())]),
                "StandardScaler then LinearDiscriminantAnalysis",
            ),
            "FeatureUnion": (
                Pipeline([
                    ("standardize", StandardScaler()),
                    ("features", FeatureUnion([
                        ("pca", PCA(n_components=self.selectedCount, random_state=self.randomSeed)),
                        ("kBest", SelectKBest(f_classif, k=self.selectedCount)),
                    ])),
                    ("logistic", self.logisticModel()),
                ]),
                f"StandardScaler, then PCA ( {self.selectedCount} ) joined with SelectKBest f_classif ( {self.selectedCount} ), then logistic regression",
            ),
        }

        # Score Each Workflow Inside The Training Windows And On The Test Windows
        groupSplitter = GroupKFold(n_splits=self.groupFolds)
        workflowRows = []
        for workflowName, (workflowModel, workflowSetting) in workflows.items():
            foldScores = cross_val_score(workflowModel, self.trainFeatures, self.trainLabels, groups=self.trainSubjects, cv=groupSplitter, scoring="f1_macro", n_jobs=self.nJobs)
            testScores = self.fitAndScore(workflowModel, self.trainFeatures, self.testFeatures)
            workflowRows.append({
                "workflow": workflowName, "setting": workflowSetting,
                "groupFoldMacroF1Mean": foldScores.mean(), "groupFoldMacroF1Std": foldScores.std(ddof=1),
                "testAccuracy": testScores["testAccuracy"], "testMacroF1": testScores["testMacroF1"],
            })
        workflowTable = pd.DataFrame(workflowRows)
        writeCsv(workflowTable, self.resultsDirectory / "pipelines.csv")
        return workflowTable
