# This file is responsible for writing every setting, definition and library version of the run into one parameter table
# %%
# Importing Libraries
import platform
import re

import hmmlearn
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from common.csvWriters import writeCsv


# %%
# Parameter Steps
class parameterSteps:
    def writeParameters(self, variants):
        """
        This method writes every setting, definition and library version of the run

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        Dataframe of the recorded parameters
        """
        # Collect The Rows Of Every Group In The Order Of The Table
        parameterRows = [
            *self.listRunRows(), *self.listDataRows(variants), *self.listFeatureRows(variants), *self.listClassifierRows(), *self.listProtocolRows(),
        ]
        parameterTable = pd.DataFrame(
            [{"section": sectionName, "parameter": parameterName, "value": parameterValue} for sectionName, parameterName, parameterValue in parameterRows]
        )
        writeCsv(parameterTable, self.resultsDirectory / "runParameters.csv")
        return parameterTable

    def listRunRows(self):
        """
        This method lists the seed, the worker count and the library versions of the run

        Arguments
        =========
        None

        Output
        ======
        List of tuples: section, parameter and value
        """
        # Record The Run Settings
        return [
            ("run", "randomSeed", self.randomSeed), ("run", "nJobs", self.nJobs), ("run", "python", platform.python_version()),
            ("run", "numpy", np.__version__), ("run", "pandas", pd.__version__), ("run", "scipy", scipy.__version__),
            ("run", "scikit-learn", sklearn.__version__), ("run", "hmmlearn", hmmlearn.__version__), ("run", "matplotlib", matplotlib.__version__),
        ]

    def listDataRows(self, variants):
        """
        This method lists the recording layout, the downsampling, the segment rule and the row counts of the data

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        List of tuples: section, parameter and value
        """
        # Name The Placements In Words, Joined As Running Text
        placementWords = [re.sub("([A-Z])", " \\1", placementName).lower() for placementName in self.loader.placementColumns]
        placementText = ", ".join(placementWords[:-1]) + " and " + placementWords[-1]

        # Record The Data Settings
        return [
            ("data", "subjects", len(self.loader.subjects)), ("data", "sourceRateHz", self.loader.sourceRate), ("data", "targetRateHz", self.loader.targetRate),
            ("data", "signals", " ".join(self.loader.signalNames)),
            ("data", "placements", placementText),
            ("data", "decimation", f"scipy.signal.decimate by {self.loader.decimationFactor}: an order {self.loader.filterOrder} Chebyshev type I low pass run forward and backward over each whole recording, then one sample in {self.loader.decimationFactor} with its label"),
            ("data", "segments", f"runs of one label in the {self.loader.targetRate:g} Hz timeline, split at every label change; the runs without activity are then dropped, so a pause without activity ends a segment"),
            ("data", "rawSampleFraction", self.rawSampleFraction),
            ("data", "rawSampleRule", "the same seeded share of the samples of every participant and activity, kept in time order and shared by all protocols and supervised classifiers"),
            ("data", "rawSubsampleRows", len(variants["rawSubsample"]["rows"])), ("data", "rawFullRows", len(variants["rawFull"]["rows"])),
        ]

    def listFeatureRows(self, variants):
        """
        This method lists the window length, the overlap and the definition of every feature

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        List of tuples: section, parameter and value
        """
        # Record The Feature Settings
        extractor = self.extractor
        return [
            ("features", "windowSamples", extractor.windowSamples),
            ("features", "windowSteps", " and ".join(
                f"{stepSamples} samples for {variantName} ({100 * (1 - stepSamples / extractor.windowSamples):g}% overlap)" for variantName, stepSamples in self.windowSteps.items()
            )),
            *[("features", f"overlap_{variantName}", 100 * (1 - stepSamples / extractor.windowSamples)) for variantName, stepSamples in self.windowSteps.items()],
            ("features", "windowRule", "windows lie inside one activity segment, so none mixes two activities or crosses a dropped gap"),
            ("features", "featureCount", variants["features80"]["features"].shape[1]),
            ("features", "zeroCrossings", "sign changes of the window after its mean is removed, so gravity does not hide them"),
            ("features", "peakToPeakAndRange", "both are the maximum minus the minimum of the window, so the two features are identical and the forest importance splits between them"),
            ("features", "fftDcComponent", "magnitude of the zero frequency bin divided by the window length, the absolute window mean"),
            ("features", "spectralEnergy", "sum of the squared magnitudes of the one sided bins without the zero frequency bin, divided by the window length"
             + (f"; with the odd window length of {extractor.windowSamples} samples Parseval's theorem makes this exactly {extractor.windowSamples / 2:g} times the population variance, so after z-scoring it is the same feature as the variance and the forest importance splits between them"
                if extractor.windowSamples % 2 else "")),
            ("features", "spectralEntropy", "entropy in bits of the one sided bin powers without the zero frequency bin, normalised to sum to one"),
            ("features", "wavelet", f"orthonormal Haar wavelet, {extractor.waveletLevels} levels, an odd length extended by repeating the last value; the three features use the detail coefficients of all levels"),
            ("features", "waveletFeatures", "sum of the detail coefficients, the square of that sum, and the sum of the squared coefficients as the energy"),
            ("features", "correlations", "Pearson correlation of each pair of axes of each accelerometer within the window, zero for a flat axis"),
            ("features", "norm", "mean and population variance of the Euclidean norm of each accelerometer within the window"),
        ]
