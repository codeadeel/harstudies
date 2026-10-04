# This file is responsible for applying the scaling, normalization, binarization, standardization and label encoding transformers
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer, LabelEncoder, MinMaxScaler, Normalizer, StandardScaler

from common.csvWriters import writeCsv


# %%
# Preparation Sections
class preparationSections:
    def prepareData(self):
        """
        This method applies the scaling, normalization, binarization, standardization and label encoding transformers to the training windows and measures their effect ( chapter Preparing Data )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one check per row, measured before and after the transformer
        """
        # Scale Every Feature To The Unit Range
        scaledFeatures = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.trainFeatures)
        preparationRows = [
            {"step": "scaling", "transformer": "MinMaxScaler", "setting": "feature_range (0, 1)", "check": "smallest value", "before": self.trainFeatures.min(), "after": scaledFeatures.min()},
            {"step": "scaling", "transformer": "MinMaxScaler", "setting": "feature_range (0, 1)", "check": "largest value", "before": self.trainFeatures.max(), "after": scaledFeatures.max()},
        ]

        # Normalise Every Window To Unit Length
        normalisedFeatures = Normalizer(norm="l2").fit_transform(self.trainFeatures)
        rowNormsBefore = np.linalg.norm(self.trainFeatures, axis=1)
        rowNormsAfter = np.linalg.norm(normalisedFeatures, axis=1)
        preparationRows += [
            {"step": "normalization", "transformer": "Normalizer", "setting": "norm l2", "check": "smallest window norm", "before": rowNormsBefore.min(), "after": rowNormsAfter.min()},
            {"step": "normalization", "transformer": "Normalizer", "setting": "norm l2", "check": "largest window norm", "before": rowNormsBefore.max(), "after": rowNormsAfter.max()},
        ]

        # Binarise At The Threshold
        binaryFeatures = Binarizer(threshold=self.binarizerThreshold).fit_transform(self.trainFeatures)
        preparationRows.append({
            "step": "binarization", "transformer": "Binarizer", "setting": f"threshold {self.binarizerThreshold:g}",
            "check": "share of values above the threshold, then share of ones",
            "before": float(np.mean(self.trainFeatures > self.binarizerThreshold)), "after": float(binaryFeatures.mean()),
        })

        # Standardise Every Feature
        standardFeatures = StandardScaler().fit_transform(self.trainFeatures)
        preparationRows += [
            {"step": "standardization", "transformer": "StandardScaler", "setting": "default", "check": "largest absolute feature mean",
             "before": np.abs(self.trainFeatures.mean(axis=0)).max(), "after": np.abs(standardFeatures.mean(axis=0)).max()},
            {"step": "standardization", "transformer": "StandardScaler", "setting": "default", "check": "smallest feature standard deviation",
             "before": self.trainFeatures.std(axis=0).min(), "after": standardFeatures.std(axis=0).min()},
            {"step": "standardization", "transformer": "StandardScaler", "setting": "default", "check": "largest feature standard deviation",
             "before": self.trainFeatures.std(axis=0).max(), "after": standardFeatures.std(axis=0).max()},
        ]

        # Encode The Activity Names As Integers
        labelEncoder = LabelEncoder().fit(self.trainLabels)
        for classCode, className in enumerate(labelEncoder.classes_):
            preparationRows.append({
                "step": "label encoding", "transformer": "LabelEncoder", "setting": "classes in sorted order",
                "check": f"code of {className}", "before": np.nan, "after": int(labelEncoder.transform([className])[0]),
            })
            if labelEncoder.transform([className])[0] != classCode:
                raise ValueError(f"LabelEncoder gave {className} an unexpected code")

        # Write The Preparation Table
        preparationTable = pd.DataFrame(preparationRows)
        writeCsv(preparationTable, self.resultsDirectory / "dataPreparation.csv")
        return preparationTable
