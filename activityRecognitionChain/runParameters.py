# This file is responsible for writing every recorded setting and check value of the run to runParameters.csv
# %%
# Importing Libraries
import platform

import hmmlearn
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from common.csvWriters import writeCsv


# %%
# Run Parameters
class runParameters:
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
        # Collect The Protocol, Feature And Classifier Settings
        protocolRows = self.protocolSettingRows() + self.featureSettingRows() + self.classifierSettingRows()

        # Add The Run Environment In Front And The Recorded Checks Behind
        environmentRows = self.environmentSettingRows()
        parameterTable = pd.DataFrame(
            [{"section": sectionName, "parameter": parameterName, "value": parameterValue} for sectionName, parameterName, parameterValue in environmentRows + protocolRows]
            + self.parameterRows
        )
        writeCsv(parameterTable, self.resultsDirectory / "runParameters.csv")
        return parameterTable

    def environmentSettingRows(self):
        """
        This method lists the seed, the worker count and the library versions of the run

        Arguments
        =========
        None

        Output
        ======
        List of ( section , parameter , value ) tuples
        """
        # List The Run Settings
        return [
            ("run", "randomSeed", self.randomSeed), ("run", "nJobs", self.nJobs), ("run", "python", platform.python_version()),
            ("run", "numpy", np.__version__), ("run", "pandas", pd.__version__), ("run", "scipy", scipy.__version__),
            ("run", "scikit-learn", sklearn.__version__), ("run", "hmmlearn", hmmlearn.__version__), ("run", "matplotlib", matplotlib.__version__),
        ]

    def protocolSettingRows(self):
        """
        This method lists the evaluation protocol that every section shares

        Arguments
        =========
        None

        Output
        ======
        List of ( section , parameter , value ) tuples
        """
        # List The Protocol Settings
        return [
            ("protocol", "windowSeconds", self.windowSeconds), ("protocol", "stepSeconds", self.stepSeconds),
            ("protocol", "toolboxStepSeconds", self.toolboxStepSeconds), ("protocol", "toolboxSweepStepSeconds", self.toolboxSweepStepSeconds),
            ("protocol", "toolboxStepSamples", self.evaluator.secondsToSamples(self.toolboxStepSeconds)),
            ("protocol", "toolboxSweepStepSamples", self.evaluator.secondsToSamples(self.toolboxSweepStepSeconds)),
            ("protocol", "sweepStepRule", f"the step equals the window below {self.stepSeconds:g} s and is {self.stepSeconds:g} s from there up"),
            ("protocol", "windowLabelRule", "majority label of the window frames, a tie going to the centre frame's label when that label is among the tied ones and otherwise to the lowest tied label"),
            ("protocol", "frameMappingRule", "every frame takes the prediction of the test window whose geometric centre (start plus half of the window length minus one) is nearest, ties to the earlier window; block membership, label ties and event scoring use the centre frame (start plus half the window length, rounded down)"),
            ("protocol", "featureScaling", "z-score with the training windows of the round"),
            ("protocol", "zeroCrossingLine", "mean of the channel over the whole recording, test frames included (in the person-independent scheme the test participant's own recording); it uses no labels"),
            ("protocol", "purging", "training windows that share frames with the held out windows or with the held out block are dropped"),
            ("protocol", "roundAveraging", "metrics are computed per cross validation round and averaged over rounds"),
            ("protocol", "precisionConvention", "a class never predicted in a round has precision zero; the Defined columns leave it out instead"),
            ("protocol", "sweepWindowSeconds", ", ".join(f"{windowValue:g}" for windowValue in self.sweepWindowSeconds)),
            ("protocol", "maximumSweepWindowSeconds", max(self.sweepWindowSeconds)),
            ("protocol", "selectedFeatureCounts", ", ".join(str(featureCount) for featureCount in self.featureCounts)),
        ]
