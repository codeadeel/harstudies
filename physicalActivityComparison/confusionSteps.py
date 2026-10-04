# This file is responsible for writing and plotting the pooled shuffled fold confusions of the classifiers that Attal et al. ( 2015 ) print
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import saveFigure


# %%
# Confusion Steps
class confusionSteps:
    def plotConfusions(self, pooledConfusions):
        """
        This method writes and plots the pooled shuffled fold confusions of k-NN and the HMM for raw data and features, as in Attal et al.'s Tables 5, 6, 9 and 10

        Arguments
        =========
        pooledConfusions : Pooled confusion counts per variant, protocol and classifier

        Output
        ======
        Dataframe of the confusion counts in long form
        """
        # Take The Four Confusions Attal Et Al. Print
        shownConfusions = [
            ("rawSubsample", "kNearestNeighbour", "5"), ("rawFull", "hiddenMarkovModel", "6"),
            ("features80", "kNearestNeighbour", "9"), ("features80", "hiddenMarkovModel", "10"),
        ]
        confusionRows = []
        figure, axes = plt.subplots(2, 2, figsize=(16, 14))
        classNames = [self.loader.classNames[classLabel] for classLabel in self.loader.classLabels]
        for panelAxis, (variantName, classifierName, tableNumber) in zip(axes.ravel(), shownConfusions):
            confusionCounts = pooledConfusions[(variantName, "shuffledFolds", classifierName)]
            rowShares = 100 * confusionCounts / confusionCounts.sum(axis=1, keepdims=True)
            confusionRows += self.listConfusionRows(confusionCounts, classNames, variantName, classifierName, tableNumber)
            self.drawConfusion(panelAxis, rowShares, classNames, variantName, classifierName, tableNumber)
            self.writeConfusionLayout(rowShares, classNames, tableNumber)
        figure.suptitle("Pooled confusion over the shuffled folds")
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "confusionMatrices.png")
        confusionTable = pd.DataFrame(confusionRows)
        writeCsv(confusionTable, self.resultsDirectory / "confusionMatrices.csv")
        return confusionTable

    def listConfusionRows(self, confusionCounts, classNames, variantName, classifierName, tableNumber):
        """
        This method lists the pooled confusion counts of one classifier in long form

        Arguments
        =========
        confusionCounts : Pooled confusion counts ( true classes as rows )
        classNames : Activity names in label order
        variantName : Name of the variant
        classifierName : Name of the classifier
        tableNumber : Number of the table of Attal et al. that shows this confusion

        Output
        ======
        List of dictionaries with one row per true and predicted class
        """
        # List One Row Per True And Predicted Class
        confusionRows = []
        for trueIndex, trueName in enumerate(classNames):
            for predictedIndex, predictedName in enumerate(classNames):
                confusionRows.append({
                    "variant": variantName, "classifier": classifierName, "paperTable": tableNumber, "trueClass": trueName,
                    "predictedClass": predictedName, "rows": int(confusionCounts[trueIndex, predictedIndex]),
                })
        return confusionRows

    def drawConfusion(self, panelAxis, rowShares, classNames, variantName, classifierName, tableNumber):
        """
        This method draws one row normalised confusion matrix with the shares written into its cells

        Arguments
        =========
        panelAxis : Panel that receives the matrix
        rowShares : Share of every true class that lands in each predicted class, in percent
        classNames : Activity names in label order
        variantName : Name of the variant
        classifierName : Name of the classifier
        tableNumber : Number of the table of Attal et al. that shows this confusion

        Output
        ======
        None
        """
        # Draw The Row Normalised Matrix
        panelAxis.imshow(rowShares, cmap="Blues", vmin=0, vmax=100)
        panelAxis.grid(False)
        for trueIndex in range(len(classNames)):
            for predictedIndex in range(len(classNames)):
                if rowShares[trueIndex, predictedIndex] >= 0.5:
                    panelAxis.text(predictedIndex, trueIndex, f"{rowShares[trueIndex, predictedIndex]:.0f}", ha="center", va="center", fontsize=6,
                                   color="white" if rowShares[trueIndex, predictedIndex] > 50 else "black")
        panelAxis.set_xticks(range(len(classNames)), classNames, rotation=45, ha="right", fontsize=7)
        panelAxis.set_yticks(range(len(classNames)), classNames, fontsize=7)
        panelAxis.set_xlabel("Predicted class")
        panelAxis.set_ylabel("True class")
        panelAxis.set_title(f"{classifierName}, {variantName} ( as Attal et al.'s Table {tableNumber}, % of true rows )", fontsize=9)

    def writeConfusionLayout(self, rowShares, classNames, tableNumber):
        """
        This method writes one confusion matrix in the layout of Attal et al.: true classes as rows and row percentages with two decimals

        Arguments
        =========
        rowShares : Share of every true class that lands in each predicted class, in percent
        classNames : Activity names in label order
        tableNumber : Number of the table of Attal et al. that shows this confusion

        Output
        ======
        None
        """
        # Write The Matrix In Attal Et Al.'s Layout: True Classes As Rows, Row Percentages With Two Decimals
        layoutTable = pd.DataFrame(np.round(rowShares, 2), columns=classNames)
        layoutTable.insert(0, "trueClass", classNames)
        writeCsv(layoutTable, self.resultsDirectory / f"confusionTable{tableNumber}.csv")
