# This file is responsible for recording the settings and check values and for writing them with the run environment
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
# Parameter Sections
class parameterSections:
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
            {"section": "run", "parameter": "matplotlib", "value": matplotlib.__version__},
        ]

        # Record The Chapter Settings
        settingRows = [
            {"section": "selection", "parameter": "keptPerMethod", "value": self.selectedCount},
            {"section": "selection", "parameter": "eliminationStep", "value": self.eliminationStep},
            {"section": "models", "parameter": "logisticIterations", "value": self.logisticIterations},
            {"section": "models", "parameter": "forestTrees", "value": self.forestTrees},
            {"section": "models", "parameter": "neighbourCount", "value": self.neighbourCount},
            {"section": "models", "parameter": "ensembleTrees", "value": self.ensembleTrees},
            {"section": "models", "parameter": "boostingRounds", "value": self.boostingRounds},
        ]

        # Write The Parameter Table
        parameterTable = pd.DataFrame(environmentRows + settingRows + self.parameterRows)
        writeCsv(parameterTable, self.resultsDirectory / "runParameters.csv")
        return parameterTable
