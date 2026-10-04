# This file is responsible for the event based scores and the loss to NULL of the basic chain, both ours and not the paper's
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Event And NULL Loss
class eventAndNullLoss:
    def evaluateEvents(self, basicTable):
        """
        This method sets the event based scores of the basic ARC next to its time based scores ( extra a )

        Arguments
        =========
        basicTable : Section 5.1 averages

        Output
        ======
        Dataframe of time based and event based scores
        """
        # Take The Paper Step Rows Of The Basic Chain
        mainRows = basicTable[(basicTable["stepVariant"] == "paperStep") & basicTable["classifier"].isin(self.basicClassifiers)]
        eventTable = mainRows[[
            "sensorSet", "classifier", "scheme", "precisionAll", "recallAll", "precisionNonNull", "recallNonNull",
            "eventPrecisionAll", "eventRecallAll", "eventPrecisionNonNull", "eventRecallNonNull", "detectedGestureSegmentShare",
        ]].reset_index(drop=True)
        writeCsv(eventTable, self.resultsDirectory / "extraEventEvaluation.csv")

        # Plot Time Based Against Event Based Scores Of 1-NN
        plotRows = eventTable[eventTable["classifier"] == "kNearestNeighbour"].reset_index(drop=True)
        figure, axis = plt.subplots(figsize=(11, 5))
        barColours = categoricalColours(4)
        groupPositions = np.arange(len(plotRows))
        for barIndex, (columnName, barLabel) in enumerate([
            ("precisionAll", "time-based precision"), ("recallAll", "time-based recall"),
            ("eventPrecisionAll", "event-based precision"), ("eventRecallAll", "event-based recall"),
        ]):
            axis.bar(groupPositions + (barIndex - 1.5) * 0.2, 100 * plotRows[columnName], width=0.2, color=barColours[barIndex], label=barLabel)
        axis.set_xticks(groupPositions, [f"{sensorSet}\n{schemeName}" for sensorSet, schemeName in zip(plotRows["sensorSet"], plotRows["scheme"])])
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent ( macro average over all classes )")
        axis.set_title("Basic ARC: time-based against event-based evaluation ( ours, not in the paper )")
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        saveFigure(figure, self.figuresDirectory / "extraEventEvaluation.png")
        return eventTable

    def sumConfusion(self, sectionName, configurationName, schemeName):
        """
        This method adds the pooled frame confusion of a configuration over both test participants

        Arguments
        =========
        sectionName : Section label
        configurationName : Configuration name within the section
        schemeName : Evaluation scheme

        Output
        ======
        Confusion counts with true classes as rows
        """
        # Add The Participants
        return sum(
            confusionCounts for (poolSection, poolConfiguration, poolScheme, _, _), confusionCounts in self.pooledConfusions.items()
            if poolSection == sectionName and poolConfiguration == configurationName and poolScheme == schemeName
        )

    def measureNullLoss(self):
        """
        This method measures the share of every gesture's frames that the basic ARC labels as NULL ( extra b )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the NULL loss per gesture
        """
        # Read The Loss To NULL From The Pooled Confusions Of 1-NN At The Paper's Step
        loader = self.evaluator.loader
        nullIndex = loader.classLabels.index(loader.nullLabel)
        lossRows = []
        for sensorSetName in self.sensorSets:
            for schemeName in self.evaluator.schemeNames:
                confusionCounts = self.sumConfusion("5.1", f"{sensorSetName}.kNearestNeighbour.paperStep", schemeName)
                for classIndex, classLabel in enumerate(loader.classLabels):
                    if classLabel == loader.nullLabel:
                        continue
                    trueFrames = confusionCounts[classIndex].sum()
                    missedFrames = trueFrames - confusionCounts[classIndex, classIndex]
                    lossRows.append({
                        "sensorSet": sensorSetName, "scheme": schemeName, "classLabel": classLabel, "className": loader.classNames[classIndex],
                        "frames": int(trueFrames), "recall": confusionCounts[classIndex, classIndex] / trueFrames,
                        "shareLostToNull": confusionCounts[classIndex, nullIndex] / trueFrames,
                        "shareOfMissesToNull": confusionCounts[classIndex, nullIndex] / missedFrames if missedFrames else 0.0,
                    })
        nullTable = pd.DataFrame(lossRows)
        writeCsv(nullTable, self.resultsDirectory / "extraNullLoss.csv")

        # Plot The Loss To NULL Per Gesture
        self.plotNullLoss(nullTable)
        return nullTable

    def plotNullLoss(self, nullTable):
        """
        This method plots the share of every gesture's frames that the basic ARC labels as NULL, one panel per scheme

        Arguments
        =========
        nullTable : Output of measureNullLoss

        Output
        ======
        Path of the saved figure
        """
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
        barColours = categoricalColours(2)
        for panelAxis, schemeName in zip(axes, self.evaluator.schemeNames):
            for barIndex, sensorSetName in enumerate(self.sensorSets):
                panelRows = nullTable[(nullTable["scheme"] == schemeName) & (nullTable["sensorSet"] == sensorSetName)]
                classPositions = np.arange(len(panelRows))
                panelAxis.bar(classPositions + (barIndex - 0.5) * 0.4, 100 * panelRows["shareLostToNull"], width=0.4, color=barColours[barIndex], label=sensorSetName)
            panelAxis.set_xticks(classPositions, panelRows["className"], rotation=30, ha="right")
            panelAxis.set_ylabel("Frames labelled NULL ( % )")
            panelAxis.set_title(f"{schemeName}: share of each gesture's frames lost to NULL ( basic ARC, 1-NN )")
            panelAxis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        figure.tight_layout()
        return saveFigure(figure, self.figuresDirectory / "extraNullLoss.png")
