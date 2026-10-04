# This file is responsible for the run setup of the smartphone exploration: recorded parameters, the shared classifier, seeds and the parameter table
# %%
# Importing Libraries
import platform

import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier

from common.csvWriters import writeCsv
from common.plotDefaults import applyPlotDefaults
from common.randomSeeds import limitThreads, seedEverything


# %%
# Run Setup
class runSetup:
    def recordParameter(self, sectionName, parameterName, parameterValue):
        """
        This method stores one setting or check value for runParameters.csv

        Arguments
        =========
        sectionName : Section the value belongs to
        parameterName : Name of the value
        parameterValue : The value itself

        Output
        ======
        None
        """
        # Keep The Parameter Row
        self.parameterRows.append({"section": sectionName, "parameter": parameterName, "value": parameterValue})

    def buildClassifier(self):
        """
        This method creates a LightGBM classifier with default hyperparameters plus classifierSettings ( seed, threads, determinism, column-wise histograms and silent logging )

        Arguments
        =========
        None

        Output
        ======
        Unfitted LightGBM classifier
        """
        # Create The Classifier
        return LGBMClassifier(**self.classifierSettings)

    def prepareRun(self):
        """
        This method fixes the seeds, the native thread pools and the plot style before the first section runs

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

    def writeParameters(self):
        """
        This method writes every recorded setting and check value together with the run environment

        Arguments
        =========
        None

        Output
        ======
        Dataframe of all recorded parameters
        """
        # Record The Run Environment
        environmentRows = [
            {"section": "run", "parameter": "randomSeed", "value": self.randomSeed},
            {"section": "run", "parameter": "nJobs", "value": self.nJobs},
            {"section": "run", "parameter": "python", "value": platform.python_version()},
            {"section": "run", "parameter": "numpy", "value": np.__version__},
            {"section": "run", "parameter": "pandas", "value": pd.__version__},
            {"section": "run", "parameter": "scipy", "value": scipy.__version__},
            {"section": "run", "parameter": "scikit-learn", "value": sklearn.__version__},
            {"section": "run", "parameter": "lightgbm", "value": lightgbm.__version__},
            {"section": "run", "parameter": "matplotlib", "value": matplotlib.__version__},
        ]

        # Write The Parameter Table
        parameterTable = pd.DataFrame(environmentRows + self.parameterRows)
        writeCsv(parameterTable, self.resultsDirectory / "runParameters.csv")
        return parameterTable
