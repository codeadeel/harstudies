# This file is responsible for estimating the step frequency of a walking signal from its spectrum and its autocorrelation
# %%
# Importing Libraries
import numpy as np
from scipy.signal import find_peaks, welch


# %%
# Cadence Estimates
class cadenceEstimates:
    def estimateSpectrum(self, boutSignals):
        """
        This method averages the Welch power spectral density of every bout, weighting each bout by its length

        Arguments
        =========
        boutSignals : Mean removed bout signals ( magnitude or vertical acceleration ), each at least one Welch segment long

        Output
        ======
        Tuple of the frequency grid and the averaged power spectral density
        """
        # Estimate One Welch Spectrum Per Bout
        boutSpectra = []
        boutWeights = []
        for boutSignal in boutSignals:
            frequencies, spectralDensity = welch(
                boutSignal, fs=self.samplingRate, window="hann", nperseg=self.welchSegmentSamples,
                noverlap=self.welchSegmentSamples // 2, nfft=self.fftLength, detrend="constant",
            )
            boutSpectra.append(spectralDensity)
            boutWeights.append(len(boutSignal))

        # Average The Bout Spectra
        return frequencies, np.average(boutSpectra, axis=0, weights=boutWeights)

    def findDominantPeak(self, frequencies, spectralDensity):
        """
        This method returns the tallest spectral peak inside the cadence band, which may be the stride or the step frequency

        Arguments
        =========
        frequencies : Spectral grid in hertz
        spectralDensity : Power spectral density on that grid

        Output
        ======
        Frequency of the tallest in band peak in hertz
        """
        # Find Local Maxima Inside The Band
        bandIndices = np.flatnonzero((frequencies >= self.lowFrequency) & (frequencies <= self.highFrequency))
        peakIndices, _ = find_peaks(spectralDensity)
        bandPeaks = np.intersect1d(peakIndices, bandIndices)
        if len(bandPeaks) == 0:
            bandPeaks = bandIndices
        return frequencies[bandPeaks[np.argmax(spectralDensity[bandPeaks])]]

    def estimateHarmonicCadence(self, frequencies, spectralDensity):
        """
        This method finds the stride frequency whose first harmonics hold the most power and returns twice it as the step frequency

        Arguments
        =========
        frequencies : Spectral grid in hertz
        spectralDensity : Power spectral density on that grid

        Output
        ======
        Tuple of the step frequency and the stride frequency in hertz
        """
        # List Stride Candidates Whose Step Frequency Lies In The Cadence Band
        candidateMask = (frequencies >= self.lowFrequency / 2) & (frequencies <= self.highFrequency / 2)
        candidateFrequencies = frequencies[candidateMask]

        # Sum The Power At The First Harmonics Of Each Candidate
        harmonicScores = np.zeros_like(candidateFrequencies)
        for harmonicNumber in range(1, self.harmonicCount + 1):
            harmonicScores += np.interp(harmonicNumber * candidateFrequencies, frequencies, spectralDensity)
        strideFrequency = candidateFrequencies[np.argmax(harmonicScores)]
        return 2 * strideFrequency, strideFrequency

    def estimateAutocorrelationCadence(self, boutMagnitudes):
        """
        This method finds the stride period as the strongest autocorrelation peak among stride lags and returns twice the stride frequency as the step frequency

        Arguments
        =========
        boutMagnitudes : Mean removed magnitude signals, each longer than the longest stride lag

        Output
        ======
        Tuple of the step frequency in hertz and the stride period in seconds
        """
        # Average The Normalised Autocorrelation Of Every Bout
        longestLag = int(np.ceil(2 * self.samplingRate / self.lowFrequency))
        boutCorrelations = []
        boutWeights = []
        for boutMagnitude in boutMagnitudes:
            fullCorrelation = np.correlate(boutMagnitude, boutMagnitude, mode="full")[len(boutMagnitude) - 1:]
            boutCorrelations.append(fullCorrelation[:longestLag + 1] / fullCorrelation[0])
            boutWeights.append(len(boutMagnitude))
        meanCorrelation = np.average(boutCorrelations, axis=0, weights=boutWeights)

        # Keep Peaks At Stride Lags Whose Step Frequency Lies In The Cadence Band
        shortestLag = 2 * self.samplingRate / self.highFrequency
        peakIndices, _ = find_peaks(meanCorrelation)
        stridePeaks = peakIndices[(peakIndices >= shortestLag) & (peakIndices <= longestLag)]
        bestPeak = stridePeaks[np.argmax(meanCorrelation[stridePeaks])]

        # Refine The Peak Lag With A Parabola Through Its Neighbours
        leftValue, centreValue, rightValue = meanCorrelation[bestPeak - 1:bestPeak + 2]
        lagOffset = 0.5 * (leftValue - rightValue) / (leftValue - 2 * centreValue + rightValue)
        strideSeconds = (bestPeak + lagOffset) / self.samplingRate
        return 2 / strideSeconds, strideSeconds
