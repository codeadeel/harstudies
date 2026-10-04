# This file is responsible for listing the cross validation rounds and splitting each one into training and test windows
# %%
# Importing Libraries
import numpy as np


# %%
# Round Splitting
class roundSplitting:
    def buildRounds(self, schemeName):
        """
        This method lists the cross validation rounds of one scheme

        Arguments
        =========
        schemeName : personDependent or personIndependent

        Output
        ======
        List of ( test subject , training subject , held out repetition block or None ) tuples
        """
        # Leave One Repetition Out Within Each Participant, Or Train On One Participant And Test On The Other
        if schemeName == "personDependent":
            return [
                (subjectNumber, subjectNumber, int(blockNumber))
                for subjectNumber in self.loader.subjects
                for blockNumber in np.unique(self.recordings[subjectNumber]["repetitions"])
            ]
        return [
            (testSubject, trainSubject, None)
            for testSubject in self.loader.subjects for trainSubject in self.loader.subjects if trainSubject != testSubject
        ]

    def splitRound(self, roundPlan, windowSamples, stepSamples):
        """
        This method selects the test windows, the training windows and the test frames of one round, dropping training windows that overlap the test windows or the held out block

        Arguments
        =========
        roundPlan : ( test subject , training subject , held out block or None )
        windowSamples : Window length in frames
        stepSamples : Step in frames

        Output
        ======
        Dictionary with the test window indices, the training window indices, the test frames and the number of purged windows
        """
        # Take Every Window Of The Other Participant, Or The Windows Centred Outside The Held Out Block
        testSubject, trainSubject, testBlock = roundPlan
        testWindows, _, testRepetitions = self.getWindows(testSubject, windowSamples, stepSamples)
        trainWindows, _, trainRepetitions = self.getWindows(trainSubject, windowSamples, stepSamples)
        if testBlock is None:
            testIndices = np.arange(len(testWindows.windowStarts))
            trainIndices = np.arange(len(trainWindows.windowStarts))
            testFrames = np.arange(testWindows.frameCount)
            return {"testIndices": testIndices, "trainIndices": trainIndices, "testFrames": testFrames, "purgedWindows": 0}

        # Drop Training Windows That Share Frames With The Test Windows Or The Held Out Block
        testIndices = np.flatnonzero(testRepetitions == testBlock)
        testFrames = np.flatnonzero(self.recordings[testSubject]["repetitions"] == testBlock)
        spanStart, spanEnd = testWindows.windowSpan(testIndices[0], testIndices[-1])
        spanStart, spanEnd = min(spanStart, int(testFrames[0])), max(spanEnd, int(testFrames[-1]) + 1)
        overlapsTest = (trainWindows.windowStarts < spanEnd) & (trainWindows.windowStarts + windowSamples > spanStart)
        outsideBlock = trainRepetitions != testBlock
        trainIndices = np.flatnonzero(outsideBlock & ~overlapsTest)
        return {
            "testIndices": testIndices, "trainIndices": trainIndices, "testFrames": testFrames,
            "purgedWindows": int(np.sum(outsideBlock & overlapsTest)),
        }
