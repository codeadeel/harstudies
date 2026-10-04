# This file is responsible for estimating the walking cadence from the raw signals and comparing it with the SSA estimate for the smartphone exploration
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from smartphoneExploration.cadenceBoutEstimates import cadenceBoutEstimates
from smartphoneExploration.cadencePlots import cadencePlots


# %%
# Cadence Sections, With The Bout Estimate And Plot Methods Inherited From Their Files
class cadenceSections(cadenceBoutEstimates, cadencePlots):
    def estimateRawCadence(self):
        """
        This method estimates each participant's cadence from the raw body acceleration and compares it with the SSA estimate ( section 6b )

        Arguments
        =========
        None

        Output
        ======
        Dataframe comparing the raw and the SSA step frequencies per participant
        """
        # Prepare The Walking Rows
        subjects = self.featureFrame["subject"].to_numpy()
        walkingMask = (self.featureFrame["Activity"] == self.walkingActivity).to_numpy()
        estimator = self.cadenceEstimator

        # Estimate The Cadence Of Every Walking Bout
        boutRows = []
        cadenceRows = []
        participantSpectrum = None
        for subject in sorted(np.unique(subjects[walkingMask]).tolist()):
            subjectMask = walkingMask & (subjects == subject)
            usedMagnitudes = []
            for boutNumber in pd.unique(self.boutNumbers[subjectMask]):
                boutRow, boutMagnitude = self.estimateBout(subject, boutNumber, subjectMask)
                if boutRow["used"]:
                    usedMagnitudes.append(boutMagnitude)
                boutRows.append(boutRow)

            # Summarise The Participant Over Its Bouts
            cadenceRow, frequencies, pooledDensity, dominantPeak = self.summariseParticipant(subject, boutRows, usedMagnitudes)
            cadenceRows.append(cadenceRow)

            # Keep The Notebook Participant's Pooled Spectrum For The Figure
            if subject == self.ssaParticipant:
                pooledCadence = estimator.estimateHarmonicCadence(frequencies, pooledDensity)[0]
                participantSpectrum = (frequencies, pooledDensity, dominantPeak, pooledCadence, cadenceRows[-1])
        boutTable = pd.DataFrame(boutRows)
        cadenceTable = pd.DataFrame(cadenceRows)
        writeCsv(boutTable, self.resultsDirectory / "cadenceBouts.csv")
        writeCsv(cadenceTable, self.resultsDirectory / "cadenceComparison.csv")

        # Record The Cadence Settings
        self.recordCadenceSettings(boutTable, cadenceTable)

        # Plot The Raw And SSA Estimates Per Participant
        self.plotCadenceComparison(cadenceTable)

        # Plot The Notebook Participant's Pooled Raw Spectrum
        self.plotRawSpectrum(participantSpectrum)
        print(f"[ SMARTPHONE : MEDIAN RAW CADENCE ] : {cadenceTable['fftStepsPerSecond'].median():.3f}")
        return cadenceTable

    def recordCadenceSettings(self, boutTable, cadenceTable):
        """
        This method records the cadence settings and the window mean gains for runParameters.csv

        Arguments
        =========
        boutTable : Dataframe of the walking bouts
        cadenceTable : Dataframe comparing the raw and the SSA step frequencies per participant

        Output
        ======
        None
        """
        estimator = self.cadenceEstimator

        # Find How Much A Window Mean Keeps Of Step And Stride Oscillations Over The Observed Cadences
        cadenceGrid = np.linspace(cadenceTable["fftStepsPerSecond"].min(), cadenceTable["fftStepsPerSecond"].max(), 1001)
        stepGain = np.max(np.abs(np.sinc(cadenceGrid * self.loader.windowSeconds)))
        strideGain = np.max(np.abs(np.sinc(cadenceGrid / 2 * self.loader.windowSeconds)))

        # Record The Cadence Settings
        self.recordParameter("cadence", "lowFrequencyHz", estimator.lowFrequency)
        self.recordParameter("cadence", "highFrequencyHz", estimator.highFrequency)
        self.recordParameter("cadence", "welchSegmentSamples", estimator.welchSegmentSamples)
        self.recordParameter("cadence", "minimumBoutSeconds", estimator.welchSegmentSamples / estimator.samplingRate)
        self.recordParameter("cadence", "fftLength", estimator.fftLength)
        self.recordParameter("cadence", "harmonicCount", estimator.harmonicCount)
        self.recordParameter("cadence", "strideCandidateLowHz", estimator.lowFrequency / 2)
        self.recordParameter("cadence", "strideCandidateHighHz", estimator.highFrequency / 2)
        self.recordParameter("cadence", "strideLagLowSeconds", 2 / estimator.highFrequency)
        self.recordParameter("cadence", "strideLagHighSeconds", 2 / estimator.lowFrequency)
        self.recordParameter("cadence", "rawSamplingRateHz", estimator.samplingRate)
        self.recordParameter("cadence", "rawNyquistHz", estimator.samplingRate / 2)
        self.recordParameter("cadence", "walkingBouts", len(boutTable))
        self.recordParameter("cadence", "usedBouts", int(boutTable["used"].sum()))
        self.recordParameter("cadence", "maxWindowMeanStepGain", float(stepGain))
        self.recordParameter("cadence", "maxWindowMeanStrideGain", float(strideGain))
