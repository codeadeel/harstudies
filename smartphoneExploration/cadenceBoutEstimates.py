# This file is responsible for estimating the cadence of single walking bouts and summarising the estimates per participant
# %%
# Importing Libraries
import numpy as np
import pandas as pd


# %%
# Cadence Bout Estimates
class cadenceBoutEstimates:
    def estimateBout(self, subject, boutNumber, subjectMask):
        """
        This method rebuilds one walking bout from its windows and estimates its cadence with three methods

        Arguments
        =========
        subject : Participant number
        boutNumber : Recording bout number
        subjectMask : Boolean array that marks the walking windows of the participant

        Output
        ======
        Tuple of the bout row and the mean removed magnitude signal of the bout
        """
        estimator = self.cadenceEstimator
        windowStepSamples = self.loader.windowStepSamples

        # Rebuild The Continuous Bout Signals
        boutIndices = np.flatnonzero(subjectMask & (self.boutNumbers == boutNumber))
        bodySignals = [estimator.stitchBout(axisWindows[boutIndices], windowStepSamples) for axisWindows in self.bodyAcceleration]
        totalSignals = [estimator.stitchBout(axisWindows[boutIndices], windowStepSamples) for axisWindows in self.totalAcceleration]
        boutMagnitude = estimator.computeMagnitude(bodySignals)
        boutRow = {
            "subject": subject, "bout": int(boutNumber), "windows": len(boutIndices),
            "seconds": len(boutMagnitude) / estimator.samplingRate,
            "used": len(boutMagnitude) >= estimator.welchSegmentSamples,
            "fftStepsPerSecond": np.nan, "acfStepsPerSecond": np.nan,
            "verticalStepsPerSecond": np.nan, "dominantPeakHz": np.nan,
        }

        # Estimate The Bout Cadence From The Magnitude With Two Methods
        if boutRow["used"]:
            frequencies, spectralDensity = estimator.estimateSpectrum([boutMagnitude])
            boutRow["fftStepsPerSecond"] = estimator.estimateHarmonicCadence(frequencies, spectralDensity)[0]
            boutRow["acfStepsPerSecond"] = estimator.estimateAutocorrelationCadence([boutMagnitude])[0]
            boutRow["dominantPeakHz"] = estimator.findDominantPeak(frequencies, spectralDensity)

            # Check The Step Rate Without Stride Doubling On The Vertical Acceleration
            verticalAcceleration = estimator.computeVerticalAcceleration(bodySignals, totalSignals)
            verticalFrequencies, verticalDensity = estimator.estimateSpectrum([verticalAcceleration])
            boutRow["verticalStepsPerSecond"] = estimator.findDominantPeak(verticalFrequencies, verticalDensity)
        return boutRow, boutMagnitude

    def summariseParticipant(self, subject, boutRows, usedMagnitudes):
        """
        This method summarises the cadence of one participant over the used walking bouts

        Arguments
        =========
        subject : Participant number
        boutRows : Bout rows collected so far
        usedMagnitudes : Magnitude signals of the used bouts of the participant

        Output
        ======
        Tuple of the table row of the participant, the pooled frequency grid, the pooled spectral density and the tallest band peak
        """
        estimator = self.cadenceEstimator

        # Summarise The Participant Over Its Bouts
        participantBouts = pd.DataFrame([boutRow for boutRow in boutRows if boutRow["subject"] == subject and boutRow["used"]])
        frequencies, pooledDensity = estimator.estimateSpectrum(usedMagnitudes)
        dominantPeak = estimator.findDominantPeak(frequencies, pooledDensity)
        fftCadence = participantBouts["fftStepsPerSecond"].median()
        acfCadence = participantBouts["acfStepsPerSecond"].median()
        verticalCadence = participantBouts["verticalStepsPerSecond"].median()
        ssaCadence = self.ssaTable.loc[self.ssaTable["subject"] == subject, "ssaStepsPerSecond"].iloc[0]
        cadenceRow = {
            "subject": subject,
            "walkingBouts": int(np.sum([boutRow["subject"] == subject for boutRow in boutRows])),
            "usedBouts": len(participantBouts),
            "usedSeconds": participantBouts["seconds"].sum(),
            "fftStepsPerSecond": fftCadence,
            "fftBoutMin": participantBouts["fftStepsPerSecond"].min(),
            "fftBoutMax": participantBouts["fftStepsPerSecond"].max(),
            "acfStepsPerSecond": acfCadence,
            "acfRelativeToFft": acfCadence / fftCadence - 1,
            "verticalStepsPerSecond": verticalCadence,
            "verticalRelativeToFft": verticalCadence / fftCadence - 1,
            "dominantPeakHz": dominantPeak,
            "dominantToStepRatio": dominantPeak / fftCadence,
            "dominantPeakIsStride": bool(abs(dominantPeak / fftCadence - 0.5) < abs(dominantPeak / fftCadence - 1.0)),
            "ssaStepsPerSecond": ssaCadence,
            "ssaToRawRatio": ssaCadence / fftCadence,
            "ssaNyquistHz": 0.5 / self.loader.windowStepSeconds,
        }
        return cadenceRow, frequencies, pooledDensity, dominantPeak
