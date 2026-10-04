# This file is responsible for cutting the activity segments into sliding windows and computing the time, frequency, wavelet, correlation and norm features of Attal et al. ( 2015 )
# %%
# Importing Libraries
import numpy as np
import pandas as pd


# %%
# Attal Feature Extractor
class attalFeatureExtractor:
    def __init__(self, signalNames, placementNames, windowSamples=25, waveletLevels=3):
        """
        This class initializes the feature extractor with the signal layout, the window length and the depth of the Haar wavelet decomposition

        Arguments
        =========
        signalNames : Names of the nine accelerometer signals in column order, three axes per placement
        placementNames : Names of the three placements in column order
        windowSamples : Window length in samples, 1 s at 25 Hz ( default : 25 )
        waveletLevels : Number of Haar decomposition levels whose detail coefficients enter the wavelet features ( default : 3 )

        Output
        ======
        None
        """
        # Store The Layout And Settings
        self.signalNames = list(signalNames)
        self.placementNames = list(placementNames)
        self.windowSamples = windowSamples
        self.waveletLevels = waveletLevels
        self.flatThreshold = 1e-12

        # Name The Features Of Every Signal In Output Order
        self.timeFeatureNames = [
            "mean", "variance", "median", "interquartileRange", "skewness", "kurtosis", "rootMeanSquare", "zeroCrossings",
            "peakToPeak", "crestFactor", "range",
        ]
        self.frequencyFeatureNames = [
            "fftDcComponent", "spectralEnergy", "spectralEntropy", "waveletSum", "waveletSquaredSum", "waveletEnergy",
        ]
        self.axisPairs = [(0, 1), (0, 2), (1, 2)]

    def cutWindows(self, sampleRows, stepSamples):
        """
        This method places sliding windows inside every activity segment, so no window mixes two activities or crosses a gap

        Arguments
        =========
        sampleRows : Sample table of mhealthLoader.sampleTable
        stepSamples : Window step in samples

        Output
        ======
        Dataframe with one row per window: subject, segment, label and the position of its first sample row
        """
        # Start A Window Every Step While It Still Fits In The Segment
        windowRows = []
        for segmentNumber, segmentRows in sampleRows.groupby("segment", sort=True):
            firstRow = int(segmentRows.index[0])
            segmentLength = len(segmentRows)
            for windowOffset in range(0, segmentLength - self.windowSamples + 1, stepSamples):
                windowRows.append({
                    "subject": int(segmentRows["subject"].iloc[0]), "segment": int(segmentNumber), "label": int(segmentRows["label"].iloc[0]),
                    "firstRow": firstRow + windowOffset,
                })
        return pd.DataFrame(windowRows)

    def haarDetails(self, windowValues):
        """
        This method computes the orthonormal Haar detail coefficients of every level, extending an odd length by repeating the last value

        Arguments
        =========
        windowValues : Array ( windows , samples ) of one signal

        Output
        ======
        Array ( windows , coefficients ) with the detail coefficients of all levels side by side
        """
        # Split Pairs Into Scaled Sums And Differences, Level By Level
        approximation = windowValues
        detailParts = []
        for _ in range(self.waveletLevels):
            if approximation.shape[1] % 2:
                approximation = np.concatenate([approximation, approximation[:, -1:]], axis=1)
            evenValues, oddValues = approximation[:, 0::2], approximation[:, 1::2]
            detailParts.append((evenValues - oddValues) / np.sqrt(2.0))
            approximation = (evenValues + oddValues) / np.sqrt(2.0)
        return np.concatenate(detailParts, axis=1)

    def signalFeatures(self, windowValues):
        """
        This method computes the eleven time domain and six frequency domain features of one signal in every window

        Arguments
        =========
        windowValues : Array ( windows , samples ) of one signal

        Output
        ======
        Array ( windows , 17 ) in the order of timeFeatureNames then frequencyFeatureNames
        """
        # Compute The Central Moments, Treating Flat Windows As Having No Shape
        windowMean = windowValues.mean(axis=1)
        centred = windowValues - windowMean[:, None]
        secondMoment = np.mean(centred ** 2, axis=1)
        isFlat = secondMoment <= self.flatThreshold
        safeSecond = np.where(isFlat, 1.0, secondMoment)
        skewness = np.where(isFlat, 0.0, np.mean(centred ** 3, axis=1) / safeSecond ** 1.5)
        kurtosis = np.where(isFlat, 0.0, np.mean(centred ** 4, axis=1) / safeSecond ** 2 - 3.0)

        # Compute The Order Statistics, Amplitudes And Crossings Of The Window Mean
        lowerQuartile, median, upperQuartile = np.percentile(windowValues, [25, 50, 75], axis=1)
        rootMeanSquare = np.sqrt(np.mean(windowValues ** 2, axis=1))
        peakToPeak = windowValues.max(axis=1) - windowValues.min(axis=1)
        crestFactor = np.divide(np.abs(windowValues).max(axis=1), rootMeanSquare, out=np.zeros_like(rootMeanSquare), where=rootMeanSquare > 0)
        zeroCrossings = np.sum(np.signbit(centred[:, 1:]) != np.signbit(centred[:, :-1]), axis=1).astype(np.float64)

        # Compute The Spectrum Features From The One Sided Bins Without The DC Bin
        windowLength = windowValues.shape[1]
        spectrum = np.fft.rfft(windowValues, axis=1)
        dcComponent = np.abs(spectrum[:, 0]) / windowLength
        binPower = np.abs(spectrum[:, 1:]) ** 2
        spectralEnergy = binPower.sum(axis=1) / windowLength
        totalPower = binPower.sum(axis=1, keepdims=True)
        powerShare = np.divide(binPower, totalPower, out=np.zeros_like(binPower), where=totalPower > 0)
        spectralEntropy = -np.sum(np.where(powerShare > 0, powerShare * np.log2(np.where(powerShare > 0, powerShare, 1.0)), 0.0), axis=1)

        # Compute The Wavelet Features From The Haar Detail Coefficients
        detailCoefficients = self.haarDetails(windowValues)
        waveletSum = detailCoefficients.sum(axis=1)
        return np.column_stack([
            windowMean, secondMoment, median, upperQuartile - lowerQuartile, skewness, kurtosis, rootMeanSquare, zeroCrossings,
            peakToPeak, crestFactor, peakToPeak.copy(),
            dcComponent, spectralEnergy, spectralEntropy, waveletSum, waveletSum ** 2, np.sum(detailCoefficients ** 2, axis=1),
        ])

    def extractFeatures(self, sampleSignals, windowTable):
        """
        This method computes the 168 features of every window: 17 per signal, 3 axis correlations and the norm mean and variance per placement

        Arguments
        =========
        sampleSignals : Array ( samples , 9 ) of sampleTable
        windowTable : Output of cutWindows

        Output
        ======
        Tuple of the feature matrix ( windows , 168 ) and the feature names
        """
        # Gather The Samples Of Every Window
        windowIndices = windowTable["firstRow"].to_numpy()[:, None] + np.arange(self.windowSamples)
        windowSignals = sampleSignals[windowIndices]

        # Compute The Features Of Every Signal
        featureParts, featureNames = [], []
        for signalIndex, signalName in enumerate(self.signalNames):
            featureParts.append(self.signalFeatures(windowSignals[:, :, signalIndex]))
            featureNames += [f"{signalName}_{featureName}" for featureName in self.timeFeatureNames + self.frequencyFeatureNames]

        # Compute The Axis Correlations And The Norm Statistics Of Every Placement
        for placementIndex, placementName in enumerate(self.placementNames):
            placementSignals = windowSignals[:, :, 3 * placementIndex:3 * placementIndex + 3]
            centred = placementSignals - placementSignals.mean(axis=1, keepdims=True)
            axisSpread = np.sqrt(np.mean(centred ** 2, axis=1))
            for firstAxis, secondAxis in self.axisPairs:
                spreadProduct = axisSpread[:, firstAxis] * axisSpread[:, secondAxis]
                covariance = np.mean(centred[:, :, firstAxis] * centred[:, :, secondAxis], axis=1)
                featureParts.append(np.divide(covariance, spreadProduct, out=np.zeros_like(covariance), where=spreadProduct > self.flatThreshold)[:, None])
                featureNames.append(f"{placementName}_correlation{'xyz'[firstAxis]}{'xyz'[secondAxis]}")
            placementNorm = np.sqrt(np.sum(placementSignals ** 2, axis=2))
            featureParts.append(np.column_stack([placementNorm.mean(axis=1), placementNorm.var(axis=1)]))
            featureNames += [f"{placementName}_normMean", f"{placementName}_normVariance"]
        return np.concatenate(featureParts, axis=1), featureNames
