# This file is responsible for loading the MHEALTH recordings, computing the features of every variant and ranking the window features once
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Data Preparation Steps
class dataPreparationSteps:
    def prepareData(self):
        """
        This method loads the recordings, lists the samples, cuts both window sets, computes their features and draws the raw subsample

        Arguments
        =========
        None

        Output
        ======
        Dictionary from variant name to its feature matrix, row table and frames per row
        """
        # Load The Recordings And List The Labelled Samples
        self.recordings = self.loader.loadAll()
        sampleSignals, sampleRows = self.loader.sampleTable(self.recordings)
        print(f"[ COMPARISON : LABELLED SAMPLES ] : {len(sampleRows)}")

        # Keep The Full Raw Samples And A Seeded Share Per Participant And Class
        keptRows = self.loader.subsampleRows(sampleRows, self.rawSampleFraction, self.randomSeed)
        variants = {
            "rawFull": {"features": sampleSignals, "rows": sampleRows, "names": self.loader.signalNames},
            "rawSubsample": {"features": sampleSignals[keptRows], "rows": sampleRows.iloc[keptRows].reset_index(drop=True), "names": self.loader.signalNames},
        }

        # Cut Both Window Sets And Compute Their Features
        for variantName, stepSamples in self.windowSteps.items():
            windowTable = self.extractor.cutWindows(sampleRows, stepSamples)
            featureMatrix, featureNames = self.extractor.extractFeatures(sampleSignals, windowTable)
            variants[variantName] = {"features": featureMatrix, "rows": windowTable, "names": featureNames}
            print(f"[ COMPARISON : WINDOWS {variantName} ] : {len(windowTable)}")
        return variants

    def rankAllWindows(self, variants):
        """
        This method ranks the features of every window once, for description only, since the evaluation ranks them inside each training fold

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        Dataframe of the ranking with the importance, the cumulative importance and the kept flag
        """
        # Rank The Features Of The Overlapping Windows With All Labels
        variant = variants["features80"]
        featureOrder, sortedImportances, keptCount = self.classifiers.rankFeatures(variant["features"], variant["rows"]["label"].to_numpy(), nJobs=self.nJobs)
        rankingTable = pd.DataFrame({
            "rank": np.arange(1, len(featureOrder) + 1), "feature": [variant["names"][featureIndex] for featureIndex in featureOrder],
            "importance": sortedImportances, "cumulativeImportance": np.cumsum(sortedImportances),
            "keptByAllWindowRanking": np.arange(len(featureOrder)) < keptCount,
        })
        writeCsv(rankingTable, self.resultsDirectory / "selectedFeatures.csv")
        return rankingTable
