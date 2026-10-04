# This file is responsible for training the configured classifier of one round and predicting its test windows
# %%
# Importing Libraries
import numpy as np
from sklearn.preprocessing import StandardScaler

from activityRecognitionChain.leftRightHmmClassifier import leftRightHmmClassifier
from activityRecognitionChain.mrmrSelector import mrmrSelector


# %%
# Round Prediction
class roundPrediction:
    def predictRound(self, configuration, roundPlan, roundSplit, windowSamples, stepSamples):
        """
        This method trains the configured classifier on the training windows and predicts the test windows

        Arguments
        =========
        configuration : Dictionary with featureType, sensorKeys, classifierName and optional featureCounts
        roundPlan : ( test subject , training subject , held out block or None )
        roundSplit : Output of splitRound
        windowSamples : Window length in frames
        stepSamples : Step in frames

        Output
        ======
        List of ( variant , predicted test window labels , extra round columns ) tuples
        """
        # Gather The Windows And Labels Of The Round
        _, trainSubject, _ = roundPlan
        _, trainLabels, trainRepetitions = self.getWindows(trainSubject, windowSamples, stepSamples)
        trainIndices = roundSplit["trainIndices"]
        classifierName = configuration["classifierName"]

        # Score Raw Sequences With One HMM Per Class
        if classifierName == "hiddenMarkovModel":
            return self.predictHmmRound(configuration, roundPlan, roundSplit, windowSamples, stepSamples)

        # Select The Feature Columns Of The Configured Sensors And Scale Them With The Training Windows
        trainFeatures, testFeatures, featureNames, featureColumns, roundColumns = self.scaleRoundFeatures(
            configuration, roundPlan, roundSplit, windowSamples, stepSamples,
        )

        # Rank Features With mRMR On The Training Windows And Evaluate Every Prefix With 1-NN
        if configuration.get("featureCounts"):
            return self.predictSelectionRound(configuration, trainFeatures, testFeatures, trainLabels[trainIndices], featureNames, featureColumns, roundColumns)

        # Fit An SVM With Its Own Parameter Search On The Training Rows
        if classifierName == "supportVectorMachine":
            svmModel, svmParameters = self.factory.fitSupportVectorMachine(trainFeatures, trainLabels[trainIndices], trainRepetitions[trainIndices])
            return [(None, svmModel.predict(testFeatures), {**roundColumns, **svmParameters})]

        # Fit Any Other Classifier Directly
        plainModel = self.factory.buildClassifier(classifierName).fit(trainFeatures, trainLabels[trainIndices])
        return [(None, plainModel.predict(testFeatures), roundColumns)]

    def predictHmmRound(self, configuration, roundPlan, roundSplit, windowSamples, stepSamples):
        """
        This method fits one HMM per class on the raw training sequences and predicts the raw test sequences, the channels scaled by the frames of the training windows

        Arguments
        =========
        configuration : Dictionary with the sensorKeys of the round
        roundPlan : ( test subject , training subject , held out block or None )
        roundSplit : Output of splitRound
        windowSamples : Window length in frames
        stepSamples : Step in frames

        Output
        ======
        List with one ( None , predicted test window labels , empty dictionary ) tuple
        """
        # Gather The Windows, Labels And Channels Of The Round
        testSubject, trainSubject, _ = roundPlan
        trainWindows, trainLabels, _ = self.getWindows(trainSubject, windowSamples, stepSamples)
        testWindows, _, _ = self.getWindows(testSubject, windowSamples, stepSamples)
        trainIndices = roundSplit["trainIndices"]
        channelIndices = self.loader.selectChannels(configuration["sensorKeys"])
        trainRecording = self.recordings[trainSubject]

        # Scale The Channels By The Frames Of The Training Windows
        trainFrames = self.findCoveredFrames(trainWindows, trainIndices, windowSamples, len(trainRecording["labels"]))
        channelScaler = StandardScaler().fit(trainRecording["data"][trainFrames][:, channelIndices])
        trainSequences = trainWindows.cutSequences(channelScaler.transform(trainRecording["data"][:, channelIndices]))[trainIndices]
        testSequences = testWindows.cutSequences(channelScaler.transform(self.recordings[testSubject]["data"][:, channelIndices]))[roundSplit["testIndices"]]

        # Fit One Model Per Class And Predict The Test Sequences
        hmmModel = leftRightHmmClassifier(randomSeed=self.randomSeed, nJobs=self.nJobs).fit(trainSequences, trainLabels[trainIndices])
        return [(None, hmmModel.predict(testSequences), {})]

    def findCoveredFrames(self, trainWindows, trainIndices, windowSamples, frameCount):
        """
        This method marks the frames that at least one training window covers

        Arguments
        =========
        trainWindows : slidingWindows object of the training recording
        trainIndices : Indices of the training windows
        windowSamples : Window length in frames
        frameCount : Number of frames in the training recording

        Output
        ======
        Boolean array with one entry per frame
        """
        # Add One At Every Window Start And Take One Away After Every Window End
        coverageSteps = np.zeros(frameCount + 1)
        np.add.at(coverageSteps, trainWindows.windowStarts[trainIndices], 1)
        np.add.at(coverageSteps, trainWindows.windowStarts[trainIndices] + windowSamples, -1)
        return np.cumsum(coverageSteps)[:-1] > 0

    def scaleRoundFeatures(self, configuration, roundPlan, roundSplit, windowSamples, stepSamples):
        """
        This method selects the feature columns of the configured sensors and scales them with the training windows

        Arguments
        =========
        configuration : Dictionary with the featureType and sensorKeys of the round
        roundPlan : ( test subject , training subject , held out block or None )
        roundSplit : Output of splitRound
        windowSamples : Window length in frames
        stepSamples : Step in frames

        Output
        ======
        Tuple of the scaled training features, the scaled test features, all feature names, the selected feature columns and the extra round columns
        """
        # Select The Feature Columns Of The Configured Sensors
        testSubject, trainSubject, _ = roundPlan
        trainIndices = roundSplit["trainIndices"]
        channelIndices = self.loader.selectChannels(configuration["sensorKeys"])
        trainFeatureMatrix, featureNames, featureChannels = self.getFeatures(trainSubject, windowSamples, stepSamples, configuration["featureType"])
        testFeatureMatrix, _, _ = self.getFeatures(testSubject, windowSamples, stepSamples, configuration["featureType"])
        featureColumns = np.flatnonzero(np.isin(featureChannels, channelIndices))

        # Scale Them With The Training Windows
        featureScaler = StandardScaler().fit(trainFeatureMatrix[trainIndices][:, featureColumns])
        trainFeatures = featureScaler.transform(trainFeatureMatrix[trainIndices][:, featureColumns])
        testFeatures = featureScaler.transform(testFeatureMatrix[roundSplit["testIndices"]][:, featureColumns])
        roundColumns = {"constantFeatures": int(np.sum(featureScaler.var_ == 0))}
        return trainFeatures, testFeatures, featureNames, featureColumns, roundColumns

    def predictSelectionRound(self, configuration, trainFeatures, testFeatures, trainLabels, featureNames, featureColumns, roundColumns):
        """
        This method ranks the features with mRMR on the training windows and predicts the test windows with 1-NN on every ranking prefix

        Arguments
        =========
        configuration : Dictionary with the featureCounts to evaluate
        trainFeatures : Scaled training features
        testFeatures : Scaled test features
        trainLabels : Training window labels
        featureNames : Name of every feature of the recording
        featureColumns : Columns of the recording features that the round uses
        roundColumns : Extra round columns shared by every variant

        Output
        ======
        List of ( feature count , predicted test window labels , extra round columns ) tuples
        """
        # Rank The Features On The Training Windows
        featureRanking = mrmrSelector(maximumFeatures=max(configuration["featureCounts"])).rankFeatures(trainFeatures, trainLabels)

        # Predict The Test Windows With Every Prefix Of The Ranking
        roundVariants = []
        for featureCount in configuration["featureCounts"]:
            selectedColumns = featureRanking[:featureCount]
            prefixModel = self.factory.buildClassifier("kNearestNeighbour").fit(trainFeatures[:, selectedColumns], trainLabels)
            topNames = [featureNames[featureColumns[columnIndex]] for columnIndex in selectedColumns[:5]]
            roundVariants.append((featureCount, prefixModel.predict(testFeatures[:, selectedColumns]), {**roundColumns, "topFeatures": " ".join(topNames)}))
        return roundVariants
