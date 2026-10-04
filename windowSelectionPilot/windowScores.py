# This file is responsible for the three training free window scores: acceleration variance, change from the previous window and spectral entropy
# %%
# Importing Libraries
import numpy as np


# %%
# Window Scores
class windowScores:
    def scoreAccelerationVariance(self, windowSignals):
        """
        This method computes score 1, the variance of the acceleration magnitude averaged over the sensors

        Arguments
        =========
        windowSignals : Windows by samples by channels array

        Output
        ======
        Score of every window
        """
        # Average The Magnitude Variance Of The Sensors
        return self.accelerationMagnitudes(windowSignals).var(axis=1).mean(axis=1)

    def scoreWindowChange(self, windowSignals, streamStarts):
        """
        This method computes score 2, the change from the previous window: the distance between the per axis acceleration means plus the distance between the per axis standard deviations, averaged over the sensors

        Arguments
        =========
        windowSignals : Windows by samples by channels array
        streamStarts : Boolean array marking the first window of every stream

        Output
        ======
        Score of every window, missing for the first window of a stream
        """
        # Summarise Every Axis Of Every Sensor
        accelerationAxes = windowSignals[:, :, self.accelerationColumns].reshape(len(windowSignals), self.windowSamples, self.sensorCount, 3)
        axisMeans = accelerationAxes.mean(axis=1)
        axisDeviations = accelerationAxes.std(axis=1)

        # Measure The Change From The Previous Window Of The Same Stream
        changeScores = np.full(len(windowSignals), np.nan)
        meanChange = np.linalg.norm(axisMeans[1:] - axisMeans[:-1], axis=2)
        deviationChange = np.linalg.norm(axisDeviations[1:] - axisDeviations[:-1], axis=2)
        changeScores[1:] = (meanChange + deviationChange).mean(axis=1)
        changeScores[streamStarts] = np.nan
        return changeScores

    def scoreSpectralEntropy(self, windowSignals):
        """
        This method computes score 3, the normalised spectral entropy of the acceleration magnitude without its mean, averaged over the sensors

        Arguments
        =========
        windowSignals : Windows by samples by channels array

        Output
        ======
        Score of every window between 0 and 1
        """
        # Normalise The Power Of The Non Zero Frequency Bins
        magnitudes = self.accelerationMagnitudes(windowSignals)
        spectrumPower = np.abs(np.fft.rfft(magnitudes - magnitudes.mean(axis=1, keepdims=True), axis=1))[:, 1:, :] ** 2
        totalPower = spectrumPower.sum(axis=1, keepdims=True)
        powerShares = np.divide(spectrumPower, totalPower, out=np.zeros_like(spectrumPower), where=totalPower > 0)

        # Take The Entropy Relative To A Flat Spectrum
        shareLogs = np.log(powerShares, out=np.zeros_like(powerShares), where=powerShares > 0)
        return (-(powerShares * shareLogs).sum(axis=1) / np.log(spectrumPower.shape[1])).mean(axis=1)
