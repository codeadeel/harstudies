# This file is responsible for the acceleration magnitudes and the recognizer features of SONAR windows
# %%
# Importing Libraries
import numpy as np


# %%
# Window Features
class windowFeatures:
    def accelerationMagnitudes(self, windowSignals):
        """
        This method computes the acceleration magnitude of every sensor

        Arguments
        =========
        windowSignals : Windows by samples by channels array

        Output
        ======
        Windows by samples by sensors array of magnitudes in m/s^2
        """
        # Take The Norm Over The Three Axes Of Each Sensor
        accelerationAxes = windowSignals[:, :, self.accelerationColumns].reshape(len(windowSignals), self.windowSamples, self.sensorCount, 3)
        return np.linalg.norm(accelerationAxes, axis=3)

    def recognizerFeatures(self, windowSignals):
        """
        This method computes the mean, the variance, the band powers and the dominant frequency of every channel

        Arguments
        =========
        windowSignals : Windows by samples by channels array

        Output
        ======
        Windows by features float32 array in the order of featureNames
        """
        # Compute The Time Domain Statistics
        channelMeans = windowSignals.mean(axis=1)
        channelVariances = windowSignals.var(axis=1)

        # Compute The One Sided Power Spectrum Of Every Channel Without Its Mean
        spectrumPower = np.abs(np.fft.rfft(windowSignals - channelMeans[:, None, :], axis=1)) ** 2 / self.windowSamples
        bandPowers = [spectrumPower[:, lowBin:highBin, :].sum(axis=1) for lowBin, highBin in self.bandRanges]
        dominantFrequencies = (np.argmax(spectrumPower[:, 1:, :], axis=1) + 1) * self.binWidth

        # Interleave The Statistics Channel By Channel
        statisticStack = np.stack([channelMeans, channelVariances, *bandPowers, dominantFrequencies], axis=2)
        return statisticStack.reshape(len(windowSignals), -1).astype(np.float32)
