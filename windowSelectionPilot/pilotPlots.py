# This file is responsible for the pilot figures: macro F1 against budget and the kept share of every activity
# %%
# Importing Libraries
import matplotlib.pyplot as plt

from common.csvWriters import readCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Pilot Plots
class pilotPlots:
    def plotBudgetCurves(self, curveTable):
        """
        This method draws macro F1 against budget per method

        Arguments
        =========
        curveTable : Output of evaluateSelections

        Output
        ======
        None
        """
        # Plot Macro F1 Against The Budget
        methodColours = dict(zip(self.methodNames, categoricalColours(len(self.methodNames))))
        figure, axis = plt.subplots(figsize=(8, 5))
        for methodName in self.methodNames:
            methodCurve = curveTable[curveTable["method"] == methodName]
            axis.errorbar(
                methodCurve["budgetPercent"], methodCurve["macroF1"] * 100, yerr=methodCurve["macroF1Std"].fillna(0) * 100,
                marker="o", markersize=4, capsize=3, color=methodColours[methodName], label=methodName,
            )
        axis.set_xscale("log")
        axis.set_xticks(self.budgetPercents, labels=[f"{budgetPercent}%" for budgetPercent in self.budgetPercents])
        axis.set_xlabel("Budget ( share of windows on which the recognizer runs )")
        axis.set_ylabel("Macro F1 over all windows ( % )")
        axis.set_title(f"Macro F1 against budget, pooled over {self.foldCount} participant folds")
        axis.legend(loc="lower right")
        saveFigure(figure, self.figuresDirectory / "macroF1VsBudget.png")

    def plotSurvival(self):
        """
        This method draws the kept share of every activity at the report budget

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Plot The Kept Share Of Every Activity At The Report Budget
        survivalTable = readCsv(self.resultsDirectory / "survivalShares.csv")
        reportTable = survivalTable[survivalTable["budgetPercent"] == self.reportPercent].pivot(index="activity", columns="method", values="keptShare")
        reportTable = reportTable.loc[self.builder.labelNames, self.methodNames] * 100
        figure, axis = plt.subplots(figsize=(9, 10))
        shareImage = axis.imshow(reportTable.to_numpy(), cmap="viridis", aspect="auto")
        for rowIndex in range(reportTable.shape[0]):
            for columnIndex in range(reportTable.shape[1]):
                axis.text(columnIndex, rowIndex, f"{reportTable.iat[rowIndex, columnIndex]:.0f}", ha="center", va="center", fontsize=7, color="white")
        axis.set_xticks(range(len(self.methodNames)), labels=self.methodNames, rotation=30, ha="right")
        axis.set_yticks(range(len(self.builder.labelNames)), labels=self.builder.labelNames, fontsize=8)
        axis.grid(False)
        figure.colorbar(shareImage, ax=axis, label="Kept share of the activity's windows ( % )")
        axis.set_title(f"Windows kept per activity at a {self.reportPercent}% budget")
        saveFigure(figure, self.figuresDirectory / "survivalAtReportBudget.png")

    def plotResults(self, curveTable):
        """
        This method draws macro F1 against budget per method and the kept share of every activity at the report budget

        Arguments
        =========
        curveTable : Output of evaluateSelections

        Output
        ======
        None
        """
        # Draw Both Figures
        self.plotBudgetCurves(curveTable)
        self.plotSurvival()
