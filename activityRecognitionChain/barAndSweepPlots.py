# This file is responsible for the bar charts, sweep curves and comparison plot that show precision and recall
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

from common.plotDefaults import categoricalColours, saveFigure


# %%
# Bar And Sweep Plots
class barAndSweepPlots:
    def plotPrecisionRecallBars(self, sectionTable, categoryColumn, panelColumns, titleText, figureName, paperTable=None):
        """
        This method draws grouped precision and recall bars per category with one panel per scheme and panel value, adding paper values as markers

        Arguments
        =========
        sectionTable : Averaged section rows
        categoryColumn : Column whose values form the bar groups
        panelColumns : Columns whose value combinations form the panels
        titleText : Figure title
        figureName : Png file name inside the figures folder
        paperTable : Optional rows with category, panel values, paperPrecision and paperRecall ( default : None )

        Output
        ======
        Path of the saved figure
        """
        # Lay Out One Panel Per Combination Of The Panel Columns
        panelKeys = list(sectionTable[panelColumns].drop_duplicates().itertuples(index=False, name=None))
        figure, axes = plt.subplots(len(panelKeys), 1, figsize=(max(9, 0.9 * sectionTable[categoryColumn].nunique() + 4), 3.4 * len(panelKeys)), squeeze=False)
        barColours = categoricalColours(2)
        for panelAxis, panelKey in zip(axes[:, 0], panelKeys):
            panelRows = sectionTable[(sectionTable[panelColumns] == pd.Series(panelKey, index=panelColumns)).all(axis=1)]
            categoryPositions = np.arange(len(panelRows))
            panelAxis.bar(categoryPositions - 0.2, 100 * panelRows["precisionAll"], width=0.4, color=barColours[0], label="precision")
            panelAxis.bar(categoryPositions + 0.2, 100 * panelRows["recallAll"], width=0.4, color=barColours[1], label="recall")

            # Mark The Paper Values Of Matching Bars
            if paperTable is not None:
                for categoryPosition, categoryValue in zip(categoryPositions, panelRows[categoryColumn]):
                    matchMask = paperTable[categoryColumn] == categoryValue
                    for columnName, columnValue in zip(panelColumns, panelKey):
                        matchMask &= paperTable[columnName] == columnValue
                    for paperRow in paperTable[matchMask].itertuples(index=False):
                        panelAxis.plot([categoryPosition - 0.2], [paperRow.paperPrecision], marker="D", color="black", markersize=6)
                        panelAxis.plot([categoryPosition + 0.2], [paperRow.paperRecall], marker="D", color="black", markersize=6)
            panelAxis.set_xticks(categoryPositions, panelRows[categoryColumn], rotation=30, ha="right")
            panelAxis.set_ylim(0, 100)
            panelAxis.set_ylabel("Percent")
            panelAxis.set_title(" · ".join(str(panelValue) for panelValue in panelKey))

        # Place The Legend Beside The First Panel
        if paperTable is not None:
            axes[0, 0].plot([], [], marker="D", color="black", linestyle="none", label="paper value")
        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        figure.suptitle(titleText)
        figure.tight_layout()
        return saveFigure(figure, self.figuresDirectory / figureName)

    def plotSweep(self, sweepTable, xColumn, xLabel, titleText, figureName, baselineRow=None, paperTable=None):
        """
        This method draws precision and recall curves over a swept setting, one panel per scheme

        Arguments
        =========
        sweepTable : Averaged rows with an optional stepVariant column
        xColumn : Swept column
        xLabel : Label of the swept axis
        titleText : Figure title
        figureName : Png file name inside the figures folder
        baselineRow : Optional dictionary from scheme to ( precision , recall ) drawn as dashed lines ( default : None )
        paperTable : Optional paper rows with scheme, xValue, paperPrecision and paperRecall ( default : None )

        Output
        ======
        Path of the saved figure
        """
        # Draw One Panel Per Scheme With A Line Per Metric And Step Variant
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        lineColours = categoricalColours(2)
        variantColumn = "stepVariant" if "stepVariant" in sweepTable.columns else None
        for panelAxis, schemeName in zip(axes, self.evaluator.schemeNames):
            schemeRows = sweepTable[sweepTable["scheme"] == schemeName]
            variantGroups = schemeRows.groupby(variantColumn, sort=False) if variantColumn else [("paperStep", schemeRows)]
            for variantName, variantRows in variantGroups:
                variantRows = variantRows.sort_values(xColumn)
                lineStyle = "-" if variantName == "paperStep" else ":"
                panelAxis.plot(variantRows[xColumn], 100 * variantRows["precisionAll"], lineStyle, marker="o", color=lineColours[0], label=f"precision, {variantName}")
                panelAxis.plot(variantRows[xColumn], 100 * variantRows["recallAll"], lineStyle, marker="s", color=lineColours[1], label=f"recall, {variantName}")
            if baselineRow is not None:
                panelAxis.axhline(100 * baselineRow[schemeName][0], color=lineColours[0], linestyle="--", linewidth=1, label="precision, mean and variance only")
                panelAxis.axhline(100 * baselineRow[schemeName][1], color=lineColours[1], linestyle="--", linewidth=1, label="recall, mean and variance only")
            if paperTable is not None:
                for paperRow in paperTable[paperTable["scheme"] == schemeName].itertuples(index=False):
                    panelAxis.plot([paperRow.xValue], [paperRow.paperPrecision], marker="D", color="black", linestyle="none")
                    panelAxis.plot([paperRow.xValue], [paperRow.paperRecall], marker="D", color="black", linestyle="none")
            panelAxis.set_xscale("log")
            sweptValues = sorted(sweepTable[xColumn].unique())
            panelAxis.set_xticks(sweptValues, [f"{sweptValue:g}" for sweptValue in sweptValues], fontsize=7, rotation=45)
            panelAxis.xaxis.set_minor_locator(NullLocator())
            panelAxis.set_xlabel(xLabel)
            panelAxis.set_ylim(0, 100)
            panelAxis.set_title(schemeName)
        axes[0].set_ylabel("Percent ( macro average over all classes )")
        if paperTable is not None:
            axes[1].plot([], [], marker="D", color="black", linestyle="none", label="paper value")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        figure.suptitle(titleText)
        figure.tight_layout()
        return saveFigure(figure, self.figuresDirectory / figureName)

    def plotPaperComparison(self, comparisonTable):
        """
        This method plots our values against the paper's printed values, one row per distinct printed value and one panel per metric

        Arguments
        =========
        comparisonTable : Output rows of comparePaper

        Output
        ======
        Path of the saved figure
        """
        mainRows = comparisonTable[(comparisonTable["stepVariant"] == "paperStep") & (comparisonTable["comparedAs"] == "main") & (comparisonTable["duplicateOf"] == "")]
        figure, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(mainRows) + 2), sharey=True)
        markerColour = categoricalColours(1)[0]
        rowPositions = np.arange(len(mainRows))[::-1]
        for panelAxis, metricName in zip(axes, ["Precision", "Recall"]):
            paperValues = mainRows[f"paper{metricName}"]
            allClassValues = mainRows[f"our{metricName}_all12ZeroPrecision"]
            panelAxis.hlines(rowPositions, paperValues, allClassValues, color="0.7", linewidth=1)
            panelAxis.scatter(paperValues, rowPositions, marker="D", color="black", zorder=3, label="paper value")
            panelAxis.scatter(allClassValues, rowPositions, color=markerColour, zorder=3, label="ours, all classes")
            panelAxis.scatter(mainRows[f"our{metricName}_nonNull11ZeroPrecision"], rowPositions, facecolors="none", edgecolors=markerColour, zorder=3, label="ours, gesture classes")
            panelAxis.set_xlim(0, 100)
            panelAxis.set_xlabel(f"{metricName} ( % )")
            panelAxis.set_title(metricName)
        axes[0].set_yticks(rowPositions, mainRows["referenceId"], fontsize=8)
        axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        figure.suptitle("Ours against the paper's printed values ( paper step, main classifier, never-predicted classes at zero precision )")
        figure.tight_layout()
        return saveFigure(figure, self.figuresDirectory / "paperComparison.png")
