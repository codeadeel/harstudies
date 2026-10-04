# This file is responsible for building the Raw, VerySimple, Simple, FFT and All feature types of the case study
# %%
# Importing Libraries
import numpy as np
from scipy.fft import dct, rfft


# %%
# Feature Extractor
class featureExtractor:
    def __init__(self, bandCount=4, cepstralCount=10, logFloor=1e-10):
        """
        This class initializes the extractor of the Raw, VerySimple, Simple, FFT and All feature types ( the toolbox's names )

        Arguments
        =========
        bandCount : Number of logarithmic FFT bands ( default : 4 )
        cepstralCount : Number of cepstral coefficients ( default : 10 )
        logFloor : Value added to the power before taking its logarithm ( default : 1e-10 )

        Output
        ======
        None
        """
        # Store The Spectral Settings
        self.bandCount = bandCount
        self.cepstralCount = cepstralCount
        self.logFloor = logFloor

    def computeMeanVariance(self, dataMatrix, windows):
        """
        This method computes the mean and the population variance of every channel in every window with cumulative sums

        Arguments
        =========
        dataMatrix : Data matrix ( frames , channels )
        windows : slidingWindows object

        Output
        ======
        Tuple of the window means and the window variances, each ( windows , channels )
        """
        # Shift Every Channel To Its Recording Mean For Numerical Stability
        shiftedData = dataMatrix - dataMatrix.mean(axis=0)
        firstSums = np.concatenate([np.zeros((1, shiftedData.shape[1])), np.cumsum(shiftedData, axis=0)])
        secondSums = np.concatenate([np.zeros((1, shiftedData.shape[1])), np.cumsum(shiftedData ** 2, axis=0)])

        # Take The Window Sums
        windowEnds = windows.windowStarts + windows.windowSamples
        shiftedMeans = (firstSums[windowEnds] - firstSums[windows.windowStarts]) / windows.windowSamples
        secondMoments = (secondSums[windowEnds] - secondSums[windows.windowStarts]) / windows.windowSamples
        windowVariances = np.maximum(secondMoments - shiftedMeans ** 2, 0.0)
        return shiftedMeans + dataMatrix.mean(axis=0), windowVariances

    def computeCrossingRates(self, windowSequences, channelCentres):
        """
        This method computes the zero crossing rate around the recording mean and the mean crossing rate around the window mean

        Arguments
        =========
        windowSequences : Window frames ( windows , window samples , channels )
        channelCentres : Recording mean of every channel, the zero line of the zero crossing rate

        Output
        ======
        Tuple of the zero crossing rates and the mean crossing rates, each ( windows , channels )
        """
        # Count Sign Changes Between Neighbouring Frames
        pairCount = max(windowSequences.shape[1] - 1, 1)
        aboveZero = windowSequences >= channelCentres[None, None, :]
        aboveMean = windowSequences >= windowSequences.mean(axis=1, keepdims=True)
        zeroCrossingRates = (aboveZero[:, 1:, :] != aboveZero[:, :-1, :]).sum(axis=1) / pairCount
        meanCrossingRates = (aboveMean[:, 1:, :] != aboveMean[:, :-1, :]).sum(axis=1) / pairCount
        return zeroCrossingRates, meanCrossingRates

    def computeSpectralFeatures(self, windowSequences):
        """
        This method computes the FFT features: power in logarithmic bands, cepstral coefficients, spectral entropy and energy

        Arguments
        =========
        windowSequences : Window frames ( windows , window samples , channels )

        Output
        ======
        Tuple of the band powers ( windows , channels , bands ), the cepstral coefficients ( windows , channels , coefficients ), the spectral entropy and the energy ( each windows , channels )
        """
        # Check The Window Is Long Enough For The Cepstrum
        windowLength = windowSequences.shape[1]
        halfLength = windowLength // 2
        if halfLength < self.cepstralCount + 1:
            raise ValueError(f"A window of {windowLength} frames is too short for {self.cepstralCount} cepstral coefficients")

        # Take The Power Spectrum Of The Mean Removed Window Without Its DC Bin
        centredSequences = windowSequences - windowSequences.mean(axis=1, keepdims=True)
        powerSpectrum = np.abs(rfft(centredSequences, axis=1)[:, 1:halfLength + 1, :]) ** 2

        # Sum The Power In Logarithmic Bands ( Upper Edges At Half, Quarter And Eighth Of The Spectrum )
        binNumbers = np.arange(1, halfLength + 1)
        bandEdges = [0] + [halfLength / 2 ** power for power in range(self.bandCount - 1, 0, -1)] + [halfLength]
        bandPowers = np.stack([
            powerSpectrum[:, (binNumbers > bandEdges[bandIndex]) & (binNumbers <= bandEdges[bandIndex + 1]), :].sum(axis=1)
            for bandIndex in range(self.bandCount)
        ], axis=2)

        # Take The Cepstrum As The DCT Of The Log Power, Dropping The Zeroth Coefficient
        logPower = np.log(powerSpectrum + self.logFloor)
        cepstralCoefficients = dct(logPower, type=2, norm="ortho", axis=1)[:, 1:self.cepstralCount + 1, :].transpose(0, 2, 1)

        # Take The Entropy Of The Normalised Power And The One Sided Power Sum Divided By The Window Length
        totalPower = powerSpectrum.sum(axis=1)
        powerShares = np.divide(powerSpectrum, totalPower[:, None, :], out=np.zeros_like(powerSpectrum), where=totalPower[:, None, :] > 0)
        shareLogs = np.log2(powerShares, out=np.zeros_like(powerShares), where=powerShares > 0)
        spectralEntropy = -(powerShares * shareLogs).sum(axis=1)
        return bandPowers, cepstralCoefficients, spectralEntropy, totalPower / windowLength

    def extractFeatures(self, dataMatrix, windows, featureType, channelNames, channelCentres):
        """
        This method builds the feature matrix of one feature type for every channel, channel by channel

        Arguments
        =========
        dataMatrix : Data matrix ( frames , channels )
        windows : slidingWindows object
        featureType : One of Raw, VerySimple, Simple, FFT and All
        channelNames : Name of every channel
        channelCentres : Recording mean of every channel, used by the zero crossing rate

        Output
        ======
        Tuple of the feature matrix ( windows , features ), the feature names and the channel index of every feature
        """
        # Collect The Blocks Of Each Feature Type As ( Windows , Channels , Features ) Arrays
        featureBlocks = []
        if featureType == "Raw":
            windowSequences = windows.cutSequences(dataMatrix)
            featureBlocks.append((windowSequences.transpose(0, 2, 1), [f"raw{frameNumber}" for frameNumber in range(windows.windowSamples)]))
        if featureType in ("VerySimple", "Simple", "All"):
            windowMeans, windowVariances = self.computeMeanVariance(dataMatrix, windows)
            featureBlocks.append((np.stack([windowMeans, windowVariances], axis=2), ["mean", "variance"]))
        if featureType in ("Simple", "All"):
            zeroCrossingRates, meanCrossingRates = self.computeCrossingRates(windows.cutSequences(dataMatrix), channelCentres)
            featureBlocks.append((np.stack([zeroCrossingRates, meanCrossingRates], axis=2), ["zeroCrossingRate", "meanCrossingRate"]))
        if featureType in ("FFT", "All"):
            bandPowers, cepstralCoefficients, spectralEntropy, windowEnergy = self.computeSpectralFeatures(windows.cutSequences(dataMatrix))
            featureBlocks.append((bandPowers, [f"fftBand{bandNumber}" for bandNumber in range(1, self.bandCount + 1)]))
            featureBlocks.append((cepstralCoefficients, [f"cepstrum{coefficientNumber}" for coefficientNumber in range(1, self.cepstralCount + 1)]))
            featureBlocks.append((np.stack([spectralEntropy, windowEnergy], axis=2), ["spectralEntropy", "energy"]))

        # Stop On An Unknown Feature Type
        if not featureBlocks:
            raise ValueError(f"Unknown feature type {featureType}")

        # Lay The Features Out Channel By Channel
        featureCube = np.concatenate([featureBlock for featureBlock, _ in featureBlocks], axis=2)
        blockNames = [blockName for _, blockNames in featureBlocks for blockName in blockNames]
        featureMatrix = featureCube.reshape(featureCube.shape[0], -1)
        featureNames = [f"{blockName}_{channelName}" for channelName in channelNames for blockName in blockNames]
        featureChannels = np.repeat(np.arange(len(channelNames)), len(blockNames))
        return featureMatrix, featureNames, featureChannels
