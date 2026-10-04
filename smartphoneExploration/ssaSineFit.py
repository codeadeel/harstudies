# This file is responsible for fitting one sine wave to a reconstructed series, starting from its periodogram peak
# %%
# Importing Libraries
import numpy as np
from scipy.optimize import curve_fit


# %%
# Sine Fit
class ssaSineFit:
    def findPeriodogramPeak(self, signalValues, sampleSpacing, paddingFactor=16):
        """
        This method returns the frequency of the tallest non zero bin of a zero padded periodogram

        Arguments
        =========
        signalValues : Evenly spaced signal
        sampleSpacing : Spacing between samples in seconds
        paddingFactor : Zero padding factor of the periodogram ( default : 16 )

        Output
        ======
        Peak frequency in hertz
        """
        # Find The Tallest Bin Above Zero Frequency
        fftLength = paddingFactor * len(signalValues)
        frequencies = np.fft.rfftfreq(fftLength, d=sampleSpacing)
        periodogram = np.abs(np.fft.rfft(signalValues - signalValues.mean(), fftLength)) ** 2
        return frequencies[1:][np.argmax(periodogram[1:])]

    def sineWave(self, timeSeconds, amplitude, frequency, phase, offset):
        """
        This method evaluates a sine wave

        Arguments
        =========
        timeSeconds : Time stamps in seconds
        amplitude : Wave amplitude
        frequency : Wave frequency in hertz
        phase : Phase in radians
        offset : Constant offset

        Output
        ======
        Sine wave values at the time stamps
        """
        # Evaluate The Wave
        return amplitude * np.sin(2 * np.pi * frequency * timeSeconds + phase) + offset

    def fitSine(self, timeSeconds, signalValues, maximumFrequency, paddingFactor=16):
        """
        This method fits one sine wave to a signal, starting from the peak of its zero padded periodogram

        Arguments
        =========
        timeSeconds : Evenly spaced time stamps in seconds
        signalValues : Signal to fit
        maximumFrequency : Upper frequency bound in hertz, the Nyquist frequency of the spacing
        paddingFactor : Zero padding factor of the starting periodogram ( default : 16 )

        Output
        ======
        Dictionary with the periodogram start frequency and the fitted amplitude, frequency, phase, offset and coefficient of determination
        """
        # Start From The Periodogram Peak, Kept Inside The Frequency Bound
        sampleSpacing = timeSeconds[1] - timeSeconds[0]
        startFrequency = min(self.findPeriodogramPeak(signalValues, sampleSpacing, paddingFactor), maximumFrequency)

        # Solve The Sine, Cosine And Offset Weights At That Frequency
        angularTime = 2 * np.pi * startFrequency * timeSeconds
        designMatrix = np.column_stack([np.sin(angularTime), np.cos(angularTime), np.ones_like(timeSeconds)])
        (sineWeight, cosineWeight, startOffset), *_ = np.linalg.lstsq(designMatrix, signalValues, rcond=None)

        # Refine All Four Parameters Within The Frequency Bounds
        startParameters = [np.hypot(sineWeight, cosineWeight), startFrequency, np.arctan2(cosineWeight, sineWeight), startOffset]
        fittedParameters, _ = curve_fit(
            self.sineWave, timeSeconds, signalValues, p0=startParameters,
            bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, maximumFrequency, np.inf, np.inf]), maxfev=20000,
        )

        # Measure The Fit Quality
        fittedValues = self.sineWave(timeSeconds, *fittedParameters)
        residualSum = np.sum((signalValues - fittedValues) ** 2)
        totalSum = np.sum((signalValues - signalValues.mean()) ** 2)
        amplitude, frequency, phase, offset = fittedParameters
        return {
            "startFrequency": startFrequency,
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
            "offset": offset,
            "rSquared": 1 - residualSum / totalSum,
        }
