# This file is responsible for setting the shuffled folds and leave one participant out side by side and plotting the drop between them
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Protocol Comparison Steps
class protocolComparisonSteps:
    def compareProtocols(self, summaryTable):
        """
        This method sets the two protocols side by side for every supervised classifier and variant and measures the drop from shuffled folds to leave one participant out

        Arguments
        =========
        summaryTable : Output of summariseMetrics

        Output
        ======
        Dataframe with one row per variant and classifier
        """
        # Spread The Protocols Into Columns, With The Drops From The Shuffled Folds
        protocolTable = self.spreadProtocols(summaryTable)
        writeCsv(protocolTable, self.resultsDirectory / "protocolComparison.csv")

        # Plot Accuracy And Macro F1 Per Protocol, One Column Per Variant
        self.plotProtocolComparison(protocolTable)
        return protocolTable

    def spreadProtocols(self, summaryTable):
        """
        This method puts the metrics of each protocol in their own columns and measures the drop from the shuffled folds in percentage points

        Arguments
        =========
        summaryTable : Output of summariseMetrics

        Output
        ======
        Dataframe with one row per variant and classifier
        """
        # Spread The Protocols Into Columns
        supervisedRows = summaryTable[summaryTable["learning"] == "supervised"]
        protocolTable = None
        for protocolName in self.folds.protocolNames:
            protocolRows = supervisedRows[supervisedRows["protocol"] == protocolName][["variant", "classifier", "accuracy", "accuracyStd", "macroF1", "macroF1Std"]]
            protocolRows = protocolRows.rename(columns={columnName: f"{columnName}_{protocolName}" for columnName in ["accuracy", "accuracyStd", "macroF1", "macroF1Std"]})
            protocolTable = protocolRows if protocolTable is None else protocolTable.merge(protocolRows, on=["variant", "classifier"], how="left", sort=False)

        # Measure The Drops From The Shuffled Folds In Percentage Points, The Levels Staying Fractions
        for protocolName in self.folds.protocolNames[1:]:
            for metricName in ["accuracy", "macroF1"]:
                protocolTable[f"{metricName}DropPoints_{protocolName}"] = 100 * (protocolTable[f"{metricName}_shuffledFolds"] - protocolTable[f"{metricName}_{protocolName}"])
        return protocolTable

    def plotProtocolComparison(self, protocolTable):
        """
        This method plots accuracy, macro F1 and the drop from the shuffled folds per variant and saves the figure

        Arguments
        =========
        protocolTable : Output of spreadProtocols

        Output
        ======
        None
        """
        # Plot Accuracy And Macro F1 Per Protocol, One Column Per Variant
        variantNames = list(protocolTable["variant"].unique())
        figure, axes = plt.subplots(3, len(variantNames), figsize=(4.8 * len(variantNames), 12), sharey="row", squeeze=False)
        barColours = categoricalColours(2 * len(self.folds.protocolNames))
        for columnIndex, variantName in enumerate(variantNames):
            variantRows = protocolTable[protocolTable["variant"] == variantName]
            barPositions = np.arange(len(variantRows))
            self.drawProtocolLevels(axes, columnIndex, variantName, variantRows, barPositions, barColours)
            self.drawProtocolDrops(axes[2, columnIndex], variantName, variantRows, barPositions, barColours)
            for panelAxis in axes[:, columnIndex]:
                panelAxis.set_xticks(barPositions, variantRows["classifier"], rotation=30, ha="right", fontsize=8)
                panelAxis.set_xlim(-0.6, max(len(variantRows), 2) - 0.4)
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        axes[2, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        figure.suptitle("Shuffled folds ( P1 ) against leave one participant out ( P3 ), mean and standard deviation over folds")
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "protocolComparison.png")

    def drawProtocolLevels(self, axes, columnIndex, variantName, variantRows, barPositions, barColours):
        """
        This method draws the accuracy and the macro F1 of both protocols for one variant

        Arguments
        =========
        axes : Grid of panels of the figure
        columnIndex : Column of the grid that belongs to the variant
        variantName : Name of the variant
        variantRows : Rows of the protocol table of this variant
        barPositions : Bar position of every classifier
        barColours : Colours of the protocols and of the drops

        Output
        ======
        None
        """
        # Draw One Bar Per Protocol And Classifier With The Standard Deviation Over Folds
        for rowIndex, metricName in enumerate(["accuracy", "macroF1"]):
            panelAxis = axes[rowIndex, columnIndex]
            for protocolIndex, protocolName in enumerate(self.folds.protocolNames):
                panelAxis.bar(
                    barPositions + (protocolIndex - 0.5) * 0.4, 100 * variantRows[f"{metricName}_{protocolName}"], width=0.4,
                    yerr=100 * variantRows[f"{metricName}Std_{protocolName}"], capsize=2, color=barColours[protocolIndex], label=protocolName,
                )
            panelAxis.set_ylim(0, 100)
            panelAxis.set_title(f"{variantName}: {'accuracy' if metricName == 'accuracy' else 'macro F1'} ( % )", fontsize=9)

    def drawProtocolDrops(self, panelAxis, variantName, variantRows, barPositions, barColours):
        """
        This method draws the drop in accuracy and macro F1 from the shuffled folds to leave one participant out for one variant

        Arguments
        =========
        panelAxis : Panel that receives the bars
        variantName : Name of the variant
        variantRows : Rows of the protocol table of this variant
        barPositions : Bar position of every classifier
        barColours : Colours of the protocols and of the drops

        Output
        ======
        None
        """
        # Plot The Drops From The Shuffled Folds In Points
        for dropIndex, (columnName, dropLabel) in enumerate([
            ("accuracyDropPoints_leaveOneSubjectOut", "accuracy, P1 - P3"), ("macroF1DropPoints_leaveOneSubjectOut", "macro F1, P1 - P3"),
        ]):
            panelAxis.bar(barPositions + (dropIndex - 0.5) * 0.4, variantRows[columnName], width=0.4, color=barColours[dropIndex + len(self.folds.protocolNames)], label=dropLabel)
        panelAxis.axhline(0, color="black", linewidth=0.8)
        panelAxis.set_title(f"{variantName}: drop from shuffled folds ( points )", fontsize=9)
