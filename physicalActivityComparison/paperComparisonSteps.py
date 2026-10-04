# This file is responsible for writing the values printed by Attal et al. ( 2015 ) and setting our shuffled fold results next to them
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Paper Comparison Steps
class paperComparisonSteps:
    def buildPaperReference(self):
        """
        This method writes the values Attal et al. print in their Tables 3, 4, 7 and 8, with the F-measure recomputed from their own recall and precision

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the paper references
        """
        # Name The Columns Of The Printed Values ( Percent )
        referenceTable = pd.DataFrame(self.listPaperRows(), columns=[
            "table", "variant", "learning", "classifier", "paperAccuracy", "paperAccuracyStd", "paperF", "paperRecall", "paperPrecision", "paperSpecificity",
        ])
        referenceTable["source"] = "Attal et al. (2015), Sensors 15(12), full text XML from Europe PMC (PMC4721778)"

        # Recompute Equation 10 From The Printed Recall And Precision
        referenceTable["fFromPrintedRecallPrecision"] = (1 + self.fBeta ** 2) * referenceTable["paperRecall"] * referenceTable["paperPrecision"] / (
            self.fBeta ** 2 * referenceTable["paperPrecision"] + referenceTable["paperRecall"]
        )
        referenceTable["fMatchesWithinRounding"] = (referenceTable["fFromPrintedRecallPrecision"] - referenceTable["paperF"]).abs() <= self.fTolerance
        writeCsv(referenceTable, self.resultsDirectory / "paperReference.csv")
        return referenceTable

    def listPaperRows(self):
        """
        This method lists the values Attal et al. print in their Tables 3, 4, 7 and 8, as percent

        Arguments
        =========
        None

        Output
        ======
        List of tuples: table, variant, learning, classifier, accuracy, accuracy deviation, F, recall, precision and specificity
        """
        # Transcribe The Printed Values ( Percent ), Checked Against The Europe PMC Full Text XML
        return [
            ("3", "rawSubsample", "supervised", "kNearestNeighbour", 96.53, 0.20, 94.60, 94.57, 94.62, 99.67),
            ("3", "rawSubsample", "supervised", "randomForest", 94.89, 0.57, 82.87, 82.28, 83.46, 99.43),
            ("3", "rawSubsample", "supervised", "supportVectorMachine", 94.22, 0.28, 90.66, 90.98, 90.33, 99.56),
            ("3", "rawSubsample", "supervised", "supervisedLearningGaussianMixture", 84.54, 0.30, 69.94, 69.99, 69.88, 98.39),
            ("4", "rawFull", "unsupervised", "hiddenMarkovModel", 80.00, 2.10, 67.67, 65.02, 66.15, 97.68),
            ("4", "rawFull", "unsupervised", "kMeans", 68.42, 5.05, 49.89, 48.67, 48.55, 93.21),
            ("4", "rawFull", "unsupervised", "gaussianMixture", 73.60, 2.32, 57.68, 57.54, 58.82, 96.45),
            ("7", "features80", "supervised", "kNearestNeighbour", 99.25, 0.17, 98.85, 98.85, 98.85, 99.96),
            ("7", "features80", "supervised", "randomForest", 98.95, 0.09, 98.27, 98.24, 98.25, 99.90),
            ("7", "features80", "supervised", "supportVectorMachine", 95.55, 0.30, 93.02, 93.15, 92.90, 99.92),
            ("7", "features80", "supervised", "supervisedLearningGaussianMixture", 85.05, 0.57, 73.44, 74.44, 73.61, 99.88),
            ("8", "features80", "unsupervised", "hiddenMarkovModel", 83.89, 1.30, 69.19, 68.27, 67.74, 98.38),
            ("8", "features80", "unsupervised", "kMeans", 72.95, 2.80, 50.29, 52.20, 51.22, 97.04),
            ("8", "features80", "unsupervised", "gaussianMixture", 75.60, 1.25, 65.00, 66.29, 64.30, 97.12),
        ]

    def compareWithPaper(self, summaryTable, referenceTable):
        """
        This method places our shuffled fold results next to every value of Attal et al.'s Tables 3, 4, 7 and 8, with the differences

        Arguments
        =========
        summaryTable : Output of summariseMetrics
        referenceTable : Output of buildPaperReference

        Output
        ======
        Dataframe in Attal et al.'s layout with our values, the paper values and the differences, in percent
        """
        # Join Our Shuffled Fold Averages To The Printed Rows
        ourRows = summaryTable[summaryTable["protocol"] == "shuffledFolds"]
        comparisonTable = referenceTable.merge(ourRows, on=["variant", "learning", "classifier"], how="left")
        for ourColumn, paperColumn in [
            ("accuracy", "paperAccuracy"), ("accuracyStd", "paperAccuracyStd"), ("attalF", "paperF"), ("macroRecall", "paperRecall"),
            ("macroPrecision", "paperPrecision"), ("macroSpecificity", "paperSpecificity"),
        ]:
            comparisonTable[f"our{ourColumn[0].upper()}{ourColumn[1:]}"] = 100 * comparisonTable[ourColumn]
            comparisonTable[f"difference{ourColumn[0].upper()}{ourColumn[1:]}"] = 100 * comparisonTable[ourColumn] - comparisonTable[paperColumn]
        comparisonTable = comparisonTable[[columnName for columnName in comparisonTable.columns if columnName not in summaryTable.columns or columnName in ("variant", "learning", "classifier")]]
        writeCsv(comparisonTable, self.resultsDirectory / "attalComparison.csv")

        # Plot Our Accuracy Next To The Printed Accuracy Per Table
        self.plotPaperComparison(comparisonTable)
        return comparisonTable

    def plotPaperComparison(self, comparisonTable):
        """
        This method plots our accuracy next to the printed accuracy of every table and saves the figure

        Arguments
        =========
        comparisonTable : Output of compareWithPaper

        Output
        ======
        None
        """
        # Plot Our Accuracy Next To The Printed Accuracy Per Table
        figure, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True)
        barColours = categoricalColours(2)
        for panelAxis, (tableNumber, tableRows) in zip(axes, comparisonTable.groupby("table", sort=True)):
            barPositions = np.arange(len(tableRows))
            panelAxis.bar(barPositions - 0.2, tableRows["paperAccuracy"], width=0.4, color=barColours[0], label="Attal et al.")
            panelAxis.errorbar(barPositions - 0.2, tableRows["paperAccuracy"], yerr=tableRows["paperAccuracyStd"], fmt="none", ecolor="black", capsize=3)
            panelAxis.bar(barPositions + 0.2, tableRows["ourAccuracy"], width=0.4, color=barColours[1], label="ours, shuffled folds")
            panelAxis.errorbar(barPositions + 0.2, tableRows["ourAccuracy"], yerr=tableRows["ourAccuracyStd"], fmt="none", ecolor="black", capsize=3)
            panelAxis.set_xticks(barPositions, tableRows["classifier"], rotation=30, ha="right", fontsize=8)
            panelAxis.set_title(f"Table {tableNumber} ( {tableRows['variant'].iloc[0]}, {tableRows['learning'].iloc[0]} )", fontsize=9)
            panelAxis.set_ylim(0, 100)
        axes[0].set_ylabel("Accuracy ( % )")
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "attalComparison.png")
