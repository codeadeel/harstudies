# This file is responsible for sections 5.1 to 5.3 of the paper: the basic chain, the feature types and the window sizes
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Basic Chain Sections
class basicChainSections:
    def runBasicChain(self):
        """
        This method evaluates the basic ARC: paper windows, mean and variance, 1-NN and a true nearest class centroid at the paper's and the toolbox's step, plus the toolbox's default k at the paper step ( section 5.1 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Both Sensor Sets And Both Steps, With The Toolbox's Default k As A Sensitivity Row At The Paper Step
        sectionTables = []
        for stepName, stepSeconds in [("paperStep", self.stepSeconds), ("toolboxStep", self.toolboxStepSeconds)]:
            for sensorSetName, sensorKeys in self.sensorSets.items():
                for classifierName in self.basicClassifiers + (self.basicSensitivityClassifiers if stepName == "paperStep" else []):
                    configuration = self.buildConfiguration(sensorKeys, classifierName=classifierName, stepSeconds=stepSeconds)
                    sectionTable = self.runConfiguration("5.1", f"{sensorSetName}.{classifierName}.{stepName}", configuration)
                    sectionTable.insert(2, "sensorSet", sensorSetName)
                    sectionTable.insert(3, "stepVariant", stepName)
                    sectionTables.append(sectionTable)
        basicTable = pd.concat(sectionTables, ignore_index=True)
        writeCsv(basicTable, self.resultsDirectory / "section51BasicChain.csv")
        return basicTable

    def compareFeatureTypes(self):
        """
        This method compares the feature types with 1-NN: Raw per frame as the paper defines it, Raw concatenated over the window as a variant, VerySimple, Simple, FFT and All ( section 5.2 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Feature Set For Both Sensor Sets, Raw With One Frame Windows
        frameSeconds = 1 / self.evaluator.loader.samplingRate
        sectionTables = []
        for sensorSetName, sensorKeys in self.sensorSets.items():
            for featureSetName, featureType, isPerFrame in self.featureSets:
                configuration = self.buildConfiguration(
                    sensorKeys, featureType=featureType,
                    windowSeconds=frameSeconds if isPerFrame else None, stepSeconds=frameSeconds if isPerFrame else None,
                )
                sectionTable = self.runConfiguration("5.2", f"{sensorSetName}.{featureSetName}", configuration)
                sectionTable.insert(2, "sensorSet", sensorSetName)
                sectionTable.insert(3, "featureSet", featureSetName)
                sectionTables.append(sectionTable)
        featureTable = pd.concat(sectionTables, ignore_index=True)
        writeCsv(featureTable, self.resultsDirectory / "section52FeatureTypes.csv")
        return featureTable

    def sweepWindowSizes(self):
        """
        This method sweeps the window size with all sensors, stepping by the window below the paper step and by the paper step above, plus the toolbox's fixed step ( section 5.3 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Window Size Under Both Step Rules
        sectionTables = []
        for stepName in ["paperStep", "toolboxStep"]:
            for windowSeconds in self.sweepWindowSeconds:
                stepSeconds = min(windowSeconds, self.stepSeconds) if stepName == "paperStep" else self.toolboxSweepStepSeconds
                configuration = self.buildConfiguration(self.allSensors, windowSeconds=windowSeconds, stepSeconds=stepSeconds)
                sectionTable = self.runConfiguration("5.3", f"window{windowSeconds}.{stepName}", configuration)
                sectionTable.insert(2, "stepVariant", stepName)
                sectionTables.append(sectionTable)
        windowTable = pd.concat(sectionTables, ignore_index=True)
        windowTable["actualWindowSeconds"] = windowTable["windowSamples"] / self.evaluator.loader.samplingRate
        windowTable["actualStepSeconds"] = windowTable["stepSamples"] / self.evaluator.loader.samplingRate
        writeCsv(windowTable, self.resultsDirectory / "section53WindowSizes.csv")
        return windowTable
