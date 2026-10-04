# This file is responsible for writing and plotting the pooled confusion matrices of the basic chain
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import saveFigure


# %%
# Basic Confusions
class basicConfusions:
    def plotConfusions(self):
        """
        This method writes and plots the pooled frame confusions of the basic ARC for both schemes and both sensor sets

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the pooled confusion counts in long form
        """
        # Pool Both Participants And Normalise Every True Class Row
        confusionRows = []
        for schemeName in self.evaluator.schemeNames:
            figure, axes = plt.subplots(1, 2, figsize=(16, 7.5))
            for panelAxis, sensorSetName in zip(axes, self.sensorSets):
                confusionCounts = self.sumConfusion("5.1", f"{sensorSetName}.kNearestNeighbour.paperStep", schemeName)
                confusionRows += self.listConfusionRows(sensorSetName, schemeName, confusionCounts)
                self.drawConfusionMatrix(panelAxis, confusionCounts, sensorSetName)
            figure.suptitle(f"Basic ARC confusion, {schemeName}, both participants pooled")
            figure.tight_layout()
            saveFigure(figure, self.figuresDirectory / f"confusionBasic{schemeName[0].upper()}{schemeName[1:]}.png")
        confusionTable = pd.DataFrame(confusionRows)
        writeCsv(confusionTable, self.resultsDirectory / "basicChainConfusion.csv")
        return confusionTable

    def listConfusionRows(self, sensorSetName, schemeName, confusionCounts):
        """
        This method lists the pooled confusion counts of one sensor set and scheme in long form

        Arguments
        =========
        sensorSetName : Name of the sensor set
        schemeName : Evaluation scheme
        confusionCounts : Confusion counts with true classes as rows

        Output
        ======
        List with one row per pair of true and predicted class
        """
        # List Every Pair Of True And Predicted Class
        loader = self.evaluator.loader
        confusionRows = []
        for trueIndex, trueName in enumerate(loader.classNames):
            for predictedIndex, predictedName in enumerate(loader.classNames):
                confusionRows.append({
                    "sensorSet": sensorSetName, "scheme": schemeName, "trueClass": trueName, "predictedClass": predictedName,
                    "frames": int(confusionCounts[trueIndex, predictedIndex]),
                })
        return confusionRows

    def drawConfusionMatrix(self, panelAxis, confusionCounts, sensorSetName):
        """
        This method draws a confusion matrix with every true class row normalised to percent

        Arguments
        =========
        panelAxis : Matplotlib axis that receives the matrix
        confusionCounts : Confusion counts with true classes as rows
        sensorSetName : Name of the sensor set, used as the title

        Output
        ======
        None
        """
        # Normalise Every True Class Row
        loader = self.evaluator.loader
        rowShares = 100 * confusionCounts / confusionCounts.sum(axis=1, keepdims=True)

        # Draw The Normalised Matrix
        panelAxis.imshow(rowShares, cmap="Blues", vmin=0, vmax=100)
        panelAxis.grid(False)
        panelAxis.spines[:].set_visible(True)
        for trueIndex in range(len(loader.classNames)):
            for predictedIndex in range(len(loader.classNames)):
                cellShare = rowShares[trueIndex, predictedIndex]
                if cellShare >= 0.5:
                    panelAxis.text(predictedIndex, trueIndex, f"{cellShare:.0f}", ha="center", va="center", fontsize=6,
                                   color="white" if cellShare > 50 else "black")
        panelAxis.set_xticks(range(len(loader.classNames)), loader.classNames, rotation=45, ha="right", fontsize=7)
        panelAxis.set_yticks(range(len(loader.classNames)), loader.classNames, fontsize=7)
        panelAxis.set_xlabel("Predicted class")
        panelAxis.set_ylabel("True class")
        panelAxis.set_title(f"{sensorSetName} ( % of true frames )")
