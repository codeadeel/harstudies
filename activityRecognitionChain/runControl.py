# This file is responsible for starting a run with its seeds and plot style and finishing it with the round metrics
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv
from common.plotDefaults import applyPlotDefaults
from common.randomSeeds import limitThreads, seedEverything


# %%
# Run Control
class runControl:
    def prepareRun(self):
        """
        This method fixes the random seeds, the native thread pools and the plot style before the first section runs

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Fix The Seeds, Thread Pools And Plot Style
        seedEverything(self.randomSeed)
        limitThreads(self.nJobs)
        applyPlotDefaults()

    def writeRoundMetrics(self):
        """
        This method writes the metrics of every round of the main evaluation

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per round, scheme and variant
        """
        # Join The Rounds Of Every Section
        roundTable = pd.concat(self.roundTables, ignore_index=True).drop(columns=["topFeatures"], errors="ignore")
        writeCsv(roundTable, self.resultsDirectory / "roundMetrics.csv")
        return roundTable
