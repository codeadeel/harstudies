# This file is responsible for counting the samples, segments and rows per participant and class and plotting the labelled samples
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import saveFigure


# %%
# Inventory Steps
class inventorySteps:
    def writeInventory(self, variants):
        """
        This method counts the samples per participant and class at both rates, the dropped samples without activity, the segments and the rows of every variant

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        Dataframe of the inventory
        """
        # Count Samples And Segments Per Participant And Label, Including The Dropped Label
        inventoryTable = pd.DataFrame(self.listInventoryRows(variants))
        writeCsv(inventoryTable, self.resultsDirectory / "dataInventory.csv")

        # Plot The Samples Per Class And Participant, As In Attal Et Al.'s Figure 5
        self.plotSampleCounts(inventoryTable)
        return inventoryTable

    def listInventoryRows(self, variants):
        """
        This method counts the samples per participant and class at both rates, the segments and the rows of every variant, including the dropped label

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        List of dictionaries with one inventory row per participant and class
        """
        # Count Samples And Segments Per Participant And Label, Including The Dropped Label
        inventoryRows = []
        for subjectNumber, recording in self.recordings.items():
            segmentLabels = [segmentLabel for _, _, segmentLabel in recording["segments"]]
            for classLabel in [self.loader.nullLabel] + self.loader.classLabels:
                inventoryRow = {
                    "subject": subjectNumber, "classLabel": classLabel,
                    "className": self.loader.classNames.get(classLabel, "No activity (dropped)"),
                    "samples50Hz": int(np.sum(recording["sourceLabels"] == classLabel)), "samples25Hz": int(np.sum(recording["labels"] == classLabel)),
                    "segments": segmentLabels.count(classLabel),
                }
                for variantName, variant in variants.items():
                    variantRows = variant["rows"]
                    inventoryRow[f"rows_{variantName}"] = int(np.sum((variantRows["subject"] == subjectNumber) & (variantRows["label"] == classLabel)))
                inventoryRows.append(inventoryRow)
        return inventoryRows

    def plotSampleCounts(self, inventoryTable):
        """
        This method plots the labelled samples per activity and participant as a heat map and saves it

        Arguments
        =========
        inventoryTable : Output of listInventoryRows as a dataframe

        Output
        ======
        None
        """
        # Plot The Samples Per Class And Participant, As In Attal Et Al.'s Figure 5
        countTable = inventoryTable[inventoryTable["classLabel"] != self.loader.nullLabel].pivot(index="subject", columns="classLabel", values="samples25Hz")
        figure, axis = plt.subplots(figsize=(14, 5.5))
        countImage = axis.imshow(countTable.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
        axis.grid(False)
        textThreshold = countTable.to_numpy().max() / 2
        for rowIndex in range(countTable.shape[0]):
            for columnIndex in range(countTable.shape[1]):
                cellCount = countTable.iat[rowIndex, columnIndex]
                axis.text(columnIndex, rowIndex, f"{cellCount}", ha="center", va="center", fontsize=7, color="white" if cellCount > textThreshold else "black")
        figure.colorbar(countImage, ax=axis, label="Samples")
        axis.set_xticks(range(countTable.shape[1]), [self.loader.classNames[classLabel] for classLabel in countTable.columns], rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(countTable.shape[0]), [f"subject {subjectNumber}" for subjectNumber in countTable.index], fontsize=8)
        axis.set_title(f"Labelled samples per activity and participant at {self.loader.targetRate:g} Hz")
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "sampleCounts.png")
