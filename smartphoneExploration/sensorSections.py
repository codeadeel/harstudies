# This file is responsible for comparing the accelerometer and gyroscope importance in the walking participant model for the smartphone exploration
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Sensor Sections
class sensorSections:
    def assignSensor(self, featureName):
        """
        This method assigns a feature to the gyroscope, the accelerometer or neither by its name

        Arguments
        =========
        featureName : Unique feature name

        Output
        ======
        Sensor name: Gyro, Acc or Other
        """
        # Check Gyroscope Names First, Then Accelerometer Names
        if "Gyro" in featureName:
            return "Gyro"
        if "Acc" in featureName:
            return "Acc"
        return "Other"

    def compareWalkingSensors(self):
        """
        This method sums the walking participant model's feature importances by sensor ( section 5.3 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of importance sums and shares per sensor
        """
        # Read Both Importance Types From The Walking Model
        importanceTable = pd.DataFrame({
            "feature": self.featureNames,
            "sensor": [self.assignSensor(featureName) for featureName in self.featureNames],
            "splitImportance": self.walkingModel.booster_.feature_importance(importance_type="split"),
            "gainImportance": self.walkingModel.booster_.feature_importance(importance_type="gain"),
        })
        writeCsv(importanceTable, self.resultsDirectory / "walkingFeatureImportance.csv")

        # Sum The Importances By Sensor
        sensorTable = importanceTable.groupby("sensor").agg(
            features=("feature", "size"), splitImportance=("splitImportance", "sum"), gainImportance=("gainImportance", "sum")
        ).reindex(["Acc", "Gyro", "Other"], fill_value=0).rename_axis("sensor").reset_index()
        sensorTable["featureShare"] = sensorTable["features"] / sensorTable["features"].sum()
        sensorTable["splitShare"] = sensorTable["splitImportance"] / sensorTable["splitImportance"].sum()
        sensorTable["gainShare"] = sensorTable["gainImportance"] / sensorTable["gainImportance"].sum()
        writeCsv(sensorTable, self.resultsDirectory / "walkingSensorImportance.csv")

        # Record The Grouping Rule
        self.recordParameter("walkingSensors", "headlineImportance", "split")
        self.recordParameter("walkingSensors", "model", "5.1 WALKING participant model")

        # Plot The Shares Per Sensor
        self.plotSensorShares(sensorTable)
        return sensorTable

    def plotSensorShares(self, sensorTable):
        """
        This method draws the feature, split and gain share of every sensor

        Arguments
        =========
        sensorTable : Dataframe of importance sums and shares per sensor

        Output
        ======
        None
        """
        # Plot The Shares Per Sensor
        figure, axis = plt.subplots(figsize=(7, 5))
        barPositions = np.arange(len(sensorTable))
        shareColours = categoricalColours(3)
        for shareIndex, (shareColumn, shareLabel) in enumerate([("featureShare", "share of features"), ("splitShare", "split importance"), ("gainShare", "gain importance")]):
            axis.bar(barPositions + (shareIndex - 1) * 0.27, sensorTable[shareColumn], width=0.27, label=shareLabel, color=shareColours[shareIndex])
        axis.set_xticks(barPositions, sensorTable["sensor"])
        axis.set_ylabel("Share of the total")
        axis.set_title("Walking participant model: importance by sensor")
        axis.legend()
        saveFigure(figure, self.figuresDirectory / "walkingSensorImportance.png")
