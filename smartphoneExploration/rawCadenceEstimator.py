# This file is responsible for the raw signal cadence estimator class that joins its signal and spectrum methods
# %%
# Importing Libraries
from smartphoneExploration.cadenceEstimates import cadenceEstimates
from smartphoneExploration.cadenceSignals import cadenceSignals


# %%
# Raw Cadence Estimator, With The Signal And Estimate Methods Inherited From Their Files
class rawCadenceEstimator(cadenceSignals, cadenceEstimates):
    def __init__(self, samplingRate, lowFrequency=0.5, highFrequency=3.0, welchSegmentSamples=512, fftLength=8192, harmonicCount=2):
        """
        This class initializes the raw signal cadence estimator with its cadence band and spectral settings

        Arguments
        =========
        samplingRate : Raw sampling rate in hertz
        lowFrequency : Lowest step frequency considered in hertz ( default : 0.5 )
        highFrequency : Highest step frequency considered in hertz ( default : 3.0 )
        welchSegmentSamples : Welch segment length in samples, also the shortest bout that is used ( default : 512 )
        fftLength : Zero padded fft length that sets the spectral grid ( default : 8192 )
        harmonicCount : Number of stride harmonics summed for each stride candidate ( default : 2 )

        Output
        ======
        None
        """
        # Store The Band And Spectral Settings
        self.samplingRate = samplingRate
        self.lowFrequency = lowFrequency
        self.highFrequency = highFrequency
        self.welchSegmentSamples = welchSegmentSamples
        self.fftLength = fftLength
        self.harmonicCount = harmonicCount
