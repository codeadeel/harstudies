# This file is responsible for counting the feature groups and the activity labels for the smartphone exploration
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Exploration Sections
class explorationSections:
    def exploreFeatureGroups(self):
        """
        This method counts the features by the name before the first dash, and before the bracket for the angle features ( section 3.1 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of feature groups and their feature counts in file order
        """
        # Take The Name Before The First Dash, And Before The Bracket For The Angle Features
        groupCounts = {}
        for featureName in self.featureNames:
            groupName = featureName.split("-")[0].split("(")[0]
            groupCounts[groupName] = groupCounts.get(groupName, 0) + 1
        groupTable = pd.DataFrame({"featureGroup": list(groupCounts), "featureCount": list(groupCounts.values())})
        writeCsv(groupTable, self.resultsDirectory / "featureGroups.csv")

        # Plot The Group Sizes
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.barh(groupTable["featureGroup"], groupTable["featureCount"], color=categoricalColours(1)[0])
        axis.invert_yaxis()
        for rowIndex, featureCount in enumerate(groupTable["featureCount"]):
            axis.text(featureCount, rowIndex, f" {featureCount}", va="center", fontsize=8)
        axis.set_xlabel("Number of features")
        axis.set_title("Feature groups ( name before the first '-', angle features before '(' )")
        saveFigure(figure, self.figuresDirectory / "featureGroups.png")
        return groupTable

    def exploreActivityDistribution(self):
        """
        This method counts the windows per activity in each split ( section 3.3 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of window counts per activity and split
        """
        # Count Windows Per Activity And Split
        countTable = pd.crosstab(self.featureFrame["Activity"], self.featureFrame["Data"])
        countTable = countTable.reindex(index=self.activityOrder, columns=["Train", "Test"], fill_value=0)
        countTable["Total"] = countTable["Train"] + countTable["Test"]
        countTable["share"] = countTable["Total"] / countTable["Total"].sum()
        countTable["seconds"] = countTable["Total"] * self.loader.windowStepSeconds
        countTable = countTable.rename_axis("Activity").reset_index()
        countTable.columns.name = None
        writeCsv(countTable, self.resultsDirectory / "activityDistribution.csv")

        # Plot The Counts Per Split
        figure, axis = plt.subplots(figsize=(9, 5))
        barPositions = np.arange(len(countTable))
        splitColours = categoricalColours(2)
        for splitIndex, splitName in enumerate(["Train", "Test"]):
            axis.bar(barPositions + (splitIndex - 0.5) * 0.4, countTable[splitName], width=0.4, label=splitName, color=splitColours[splitIndex])
        axis.set_xticks(barPositions, countTable["Activity"], rotation=20, ha="right")
        axis.set_ylabel("Windows")
        axis.set_title("Activity label distribution")
        axis.legend()
        saveFigure(figure, self.figuresDirectory / "activityDistribution.png")
        return countTable
