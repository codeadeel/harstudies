# This file is responsible for running one chain configuration under both evaluation schemes and scoring it frame by frame and event by event
# %%
# Importing Libraries
import numpy as np

from activityRecognitionChain.chainClassifiers import classifierFactory
from activityRecognitionChain.featureExtractor import featureExtractor
from activityRecognitionChain.gestureLoader import gestureLoader
from activityRecognitionChain.recordingAccess import recordingAccess
from activityRecognitionChain.roundPrediction import roundPrediction
from activityRecognitionChain.roundScoring import roundScoring
from activityRecognitionChain.roundSplitting import roundSplitting


# %%
# Chain Evaluator
class chainEvaluator(recordingAccess, roundSplitting, roundScoring, roundPrediction):
    def __init__(self, dataRoot, randomSeed=42, nJobs=4, fillReplacement=None, zeroLineSubject=None):
        """
        This class initializes the evaluator with both recordings, their repetition blocks and empty window and feature caches

        Arguments
        =========
        dataRoot : Root directory that holds the ActRecTut clone
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Worker count for the SVM grid search, the k-NN queries and the parallel HMM class fits ( default : 4 )
        fillReplacement : None keeps the published values of filled frames, mean or interpolate replaces them without labels ( default : None )
        zeroLineSubject : Participant whose channel means are the zero crossing line of every recording, each recording's own means when None ( default : None )

        Output
        ======
        None
        """
        # Store The Building Blocks
        self.loader = gestureLoader(dataRoot)
        self.extractor = featureExtractor()
        self.factory = classifierFactory(randomSeed=randomSeed, nJobs=nJobs)
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.classLabels = np.asarray(self.loader.classLabels)
        self.gestureMask = self.classLabels != self.loader.nullLabel
        self.schemeNames = ["personDependent", "personIndependent"]

        # Load Both Recordings With Their Repetition Blocks And Channel Means
        self.recordings = {}
        for subjectNumber in self.loader.subjects:
            dataMatrix, frameLabels = self.loader.loadSubject(subjectNumber)

            # Replace Filled Frames With A Label Free Value When Asked
            if fillReplacement is not None:
                dataMatrix = self.replaceFilledFrames(dataMatrix, fillReplacement)
            self.recordings[subjectNumber] = {
                "data": dataMatrix,
                "labels": frameLabels,
                "repetitions": self.loader.labelRepetitions(frameLabels),
                "channelCentres": dataMatrix.mean(axis=0),
            }

        # Take The Zero Crossing Line Of Every Recording From One Participant When Asked
        if zeroLineSubject is not None:
            zeroLine = self.recordings[zeroLineSubject]["channelCentres"].copy()
            for recording in self.recordings.values():
                recording["channelCentres"] = zeroLine

        # Prepare The Caches
        self.windowCache = {}
        self.featureCache = {}

    def evaluateConfiguration(self, configuration):
        """
        This method runs every round of both schemes for one configuration and scores it frame by frame and event by event

        Arguments
        =========
        configuration : Dictionary with windowSeconds, stepSeconds, featureType, sensorKeys, classifierName and optional featureCounts and schemeNames

        Output
        ======
        Tuple of the round rows and the pooled frame confusion per ( scheme , variant , test subject )
        """
        # Convert The Window Settings To Frames
        windowSamples = self.secondsToSamples(configuration["windowSeconds"])
        stepSamples = self.secondsToSamples(configuration["stepSeconds"])

        # Run Every Round Of Both Schemes
        roundRows = []
        pooledConfusions = {}
        for schemeName in configuration.get("schemeNames", self.schemeNames):
            for roundNumber, roundPlan in enumerate(self.buildRounds(schemeName), start=1):
                testSubject, trainSubject, testBlock = roundPlan
                roundSplit = self.splitRound(roundPlan, windowSamples, stepSamples)
                testWindows, _, _ = self.getWindows(testSubject, windowSamples, stepSamples)
                testIndices = roundSplit["testIndices"]
                frameWindows = testWindows.mapFramesToWindows(roundSplit["testFrames"], testIndices[0], testIndices[-1]) - testIndices[0]
                trueFrameLabels = self.recordings[testSubject]["labels"][roundSplit["testFrames"]]

                # Score Every Variant Of The Round
                for variantName, predictedWindowLabels, extraColumns in self.predictRound(configuration, roundPlan, roundSplit, windowSamples, stepSamples):
                    frameMetrics, frameConfusion = self.scoreFrames(trueFrameLabels, predictedWindowLabels[frameWindows])
                    eventMetrics = self.scoreEvents(testSubject, roundSplit["testFrames"], testWindows.centreFrames[testIndices], predictedWindowLabels)
                    poolKey = (schemeName, variantName, testSubject)
                    pooledConfusions[poolKey] = pooledConfusions.get(poolKey, 0) + frameConfusion
                    roundRows.append({
                        "scheme": schemeName, "variant": variantName, "round": roundNumber, "testSubject": testSubject,
                        "trainSubject": trainSubject, "heldOutRepetition": -1 if testBlock is None else testBlock + 1,
                        "windowSamples": windowSamples, "stepSamples": stepSamples,
                        "trainWindows": len(roundSplit["trainIndices"]), "testWindows": len(testIndices),
                        "purgedWindows": roundSplit["purgedWindows"], "testFrames": len(roundSplit["testFrames"]),
                        **frameMetrics, **eventMetrics, **extraColumns,
                    })
        return roundRows, pooledConfusions
