# This file is responsible for writing the CPU time table and the run parameter table of the pilot
# %%
# Importing Libraries
import platform

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from common.csvWriters import writeCsv


# %%
# Pilot Tables
class pilotTables:
    def writeCpuTimes(self):
        """
        This method writes the CPU time per window of every score and of the recognizer

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per component
        """
        # Divide The Measured CPU Seconds By The Windows
        windowCount = len(self.windowTable)
        scoreSeconds = sum(self.cpuSeconds[scoreName] for scoreName in self.builder.scoreNames)
        componentSeconds = [
            ("scoring", "score 1: variance of the acceleration magnitude", self.cpuSeconds["accelerationVariance"]),
            ("scoring", "score 2: change from the previous window", self.cpuSeconds["windowChange"]),
            ("scoring", "score 3: spectral entropy", self.cpuSeconds["spectralEntropy"]),
            ("scoring", "all three scores", scoreSeconds),
            ("recognizer", "features", self.cpuSeconds["features"]),
            ("recognizer", "forest prediction", self.forestPredictionSeconds),
            ("recognizer", "features and forest prediction", self.cpuSeconds["features"] + self.forestPredictionSeconds),
        ]
        cpuTable = pd.DataFrame([
            {"part": partName, "component": componentName, "cpuSeconds": cpuValue, "windows": windowCount, "microsecondsPerWindow": cpuValue / windowCount * 1e6}
            for partName, componentName, cpuValue in componentSeconds
        ])
        writeCsv(cpuTable, self.resultsDirectory / "cpuTimes.csv")
        return cpuTable

    def writeParameters(self):
        """
        This method writes every recorded setting and count together with the run environment

        Arguments
        =========
        None

        Output
        ======
        Dataframe of all recorded parameters
        """
        # Record The Run Environment And The Settings
        parameterRows = [
            {"section": "run", "parameter": "randomSeed", "value": self.randomSeed},
            {"section": "run", "parameter": "nJobs", "value": self.nJobs},
            {"section": "run", "parameter": "python", "value": platform.python_version()},
            {"section": "run", "parameter": "numpy", "value": np.__version__},
            {"section": "run", "parameter": "pandas", "value": pd.__version__},
            {"section": "run", "parameter": "scipy", "value": scipy.__version__},
            {"section": "run", "parameter": "scikit-learn", "value": sklearn.__version__},
            {"section": "run", "parameter": "matplotlib", "value": matplotlib.__version__},
            {"section": "windows", "parameter": "stepMicroseconds", "value": self.builder.stepMicroseconds},
            {"section": "windows", "parameter": "samplingRate", "value": self.builder.samplingRate},
            {"section": "windows", "parameter": "windowSeconds", "value": self.builder.windowSeconds},
            {"section": "windows", "parameter": "windowSamples", "value": self.builder.windowSamples},
            {"section": "windows", "parameter": "gapFactor", "value": self.builder.gapFactor},
            {"section": "windows", "parameter": "bandEdges", "value": " ".join(f"{bandEdge:g}" for bandEdge in self.builder.bandEdges)},
            {"section": "windows", "parameter": "channels", "value": len(self.builder.channelNames)},
            {"section": "windows", "parameter": "features", "value": len(self.builder.featureNames)},
            {"section": "recognizer", "parameter": "forestTrees", "value": self.forestTrees},
            {"section": "recognizer", "parameter": "foldCount", "value": self.foldCount},
            {"section": "selection", "parameter": "budgetPercents", "value": " ".join(str(budgetPercent) for budgetPercent in self.budgetPercents)},
            {"section": "selection", "parameter": "reportPercent", "value": self.reportPercent},
            {"section": "selection", "parameter": "randomSeeds", "value": " ".join(str(self.randomSeed + seedOffset) for seedOffset in range(self.randomSeedCount))},
        ]

        # Write The Parameter Table
        parameterTable = pd.DataFrame(parameterRows + self.parameterRows)
        writeCsv(parameterTable, self.resultsDirectory / "runParameters.csv")
        return parameterTable
