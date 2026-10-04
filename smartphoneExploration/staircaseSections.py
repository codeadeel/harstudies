# This file is responsible for comparing the upstairs and downstairs durations of every participant for the smartphone exploration
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Staircase Sections
class staircaseSections:
    def exploreStaircase(self):
        """
        This method compares the upstairs and downstairs durations of every participant ( sections 5.4 to 5.6 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of staircase durations and the up to down ratio per participant
        """
        # Find The Staircase Durations
        staircaseTable = pd.DataFrame({
            "subject": self.durationTable.index,
            "upstairsSeconds": self.durationTable[self.upstairsActivity].to_numpy(),
            "downstairsSeconds": self.durationTable[self.downstairsActivity].to_numpy(),
        })
        staircaseTable["upDownRatio"] = staircaseTable["upstairsSeconds"] / staircaseTable["downstairsSeconds"]
        writeCsv(staircaseTable, self.resultsDirectory / "staircaseDurations.csv")

        # Plot The Durations, The Up To Down Ratio And The Duration Distribution
        stairColours = categoricalColours(2)
        self.plotStaircaseDurations(staircaseTable, stairColours)
        self.plotStaircaseRatio(staircaseTable, stairColours)
        self.plotStaircaseHistogram(staircaseTable, stairColours)
        return staircaseTable

    def plotStaircaseDurations(self, staircaseTable, stairColours):
        """
        This method draws the upstairs and downstairs seconds of every participant

        Arguments
        =========
        staircaseTable : Dataframe of staircase durations and the up to down ratio per participant
        stairColours : Colours of the upstairs and downstairs bars

        Output
        ======
        None
        """
        # Plot The Durations Per Participant
        figure, axis = plt.subplots(figsize=(12, 5))
        barPositions = np.arange(len(staircaseTable))
        axis.bar(barPositions - 0.2, staircaseTable["upstairsSeconds"], width=0.4, label="upstairs", color=stairColours[0])
        axis.bar(barPositions + 0.2, staircaseTable["downstairsSeconds"], width=0.4, label="downstairs", color=stairColours[1])
        axis.set_xticks(barPositions, staircaseTable["subject"])
        axis.set_xlabel("Participant")
        axis.set_ylabel("Seconds")
        axis.set_title("Staircase durations per participant")
        axis.legend()
        saveFigure(figure, self.figuresDirectory / "staircaseDurations.png")

    def plotStaircaseRatio(self, staircaseTable, stairColours):
        """
        This method draws the up to down ratio of every participant in ascending order

        Arguments
        =========
        staircaseTable : Dataframe of staircase durations and the up to down ratio per participant
        stairColours : Colours of the upstairs and downstairs bars

        Output
        ======
        None
        """
        # Plot The Up To Down Ratio In Ascending Order
        sortedTable = staircaseTable.sort_values(["upDownRatio", "subject"])
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.bar(np.arange(len(sortedTable)), sortedTable["upDownRatio"], color=stairColours[0])
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(sortedTable)), sortedTable["subject"])
        axis.set_xlabel("Participant ( sorted by ratio )")
        axis.set_ylabel("Upstairs seconds / downstairs seconds")
        axis.set_title("Staircase up to down duration ratio")
        saveFigure(figure, self.figuresDirectory / "staircaseRatio.png")

    def plotStaircaseHistogram(self, staircaseTable, stairColours):
        """
        This method draws the distribution of the upstairs and downstairs seconds

        Arguments
        =========
        staircaseTable : Dataframe of staircase durations and the up to down ratio per participant
        stairColours : Colours of the upstairs and downstairs bars

        Output
        ======
        None
        """
        # Plot The Duration Distribution
        allDurations = np.concatenate([staircaseTable["upstairsSeconds"], staircaseTable["downstairsSeconds"]])
        histogramBins = np.histogram_bin_edges(allDurations, bins="auto")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(staircaseTable["upstairsSeconds"], bins=histogramBins, alpha=0.6, label="upstairs", color=stairColours[0])
        axis.hist(staircaseTable["downstairsSeconds"], bins=histogramBins, alpha=0.6, label="downstairs", color=stairColours[1])
        axis.set_xlabel("Seconds per participant")
        axis.set_ylabel("Participants")
        axis.set_title("Staircase duration distribution")
        axis.legend()
        saveFigure(figure, self.figuresDirectory / "staircaseHistogram.png")
