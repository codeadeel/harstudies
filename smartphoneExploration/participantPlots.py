# This file is responsible for plotting the participant identification scores and the seconds of data per participant
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

from common.plotDefaults import categoricalColours, saveFigure


# %%
# Participant Plots
class participantPlots:
    def plotParticipantIdentification(self, identificationTable):
        """
        This method draws the identification scores next to the seconds of data per participant

        Arguments
        =========
        identificationTable : Dataframe of identification scores and durations per activity

        Output
        ======
        None
        """
        # Plot Accuracy Next To The Seconds Per Participant
        figure, (scoreAxis, durationAxis) = plt.subplots(1, 2, figsize=(13, 5))
        barPositions = np.arange(len(identificationTable))
        scoreColours = categoricalColours(3)
        scoreAxis.bar(barPositions - 0.27, identificationTable["accuracy"], width=0.27, label="accuracy, random window split", color=scoreColours[0])
        scoreAxis.bar(barPositions, identificationTable["macroF1"], width=0.27, label="macro F1, random window split", color=scoreColours[1])
        scoreAxis.bar(barPositions + 0.27, identificationTable["boutHeldOutAccuracyMean"], width=0.27, color=scoreColours[2],
                      yerr=identificationTable["boutHeldOutAccuracyStd"], capsize=3, label=f"accuracy, whole bouts held out ( mean and std over {self.boutHeldOutFolds} folds )")
        scoreAxis.set_xticks(barPositions, identificationTable["Activity"], rotation=30, ha="right")
        scoreAxis.set_ylim(0, 1)
        scoreAxis.set_title("Participant identification per activity")
        scoreAxis.legend(loc="lower left")
        durationAxis.boxplot([self.durationTable[activityName] for activityName in self.activityOrder])
        durationAxis.set_xticks(barPositions + 1, self.activityOrder, rotation=30, ha="right")
        durationAxis.set_ylabel("Seconds per participant")
        durationAxis.set_title("Seconds of data per participant")
        saveFigure(figure, self.figuresDirectory / "participantIdentification.png")

    def plotParticipantDurations(self):
        """
        This method draws a heat map of the seconds of data of every participant and activity

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Plot The Seconds Of Every Participant And Activity
        figure, axis = plt.subplots(figsize=(8, 10))
        heatImage = axis.imshow(self.durationTable.to_numpy(), aspect="auto", cmap="viridis")
        axis.grid(False)
        axis.spines[:].set_visible(True)
        axis.set_xticks(range(len(self.activityOrder)), self.activityOrder, rotation=30, ha="right")
        axis.set_yticks(range(len(self.durationTable)), self.durationTable.index)
        axis.set_ylabel("Participant")
        axis.set_title("Seconds of data per participant and activity")
        figure.colorbar(heatImage, ax=axis, label="Seconds")
        saveFigure(figure, self.figuresDirectory / "participantDurations.png")
