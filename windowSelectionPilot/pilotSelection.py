# This file is responsible for the rules that choose the windows of a stream and carry the predictions of the kept windows forward
# %%
# Importing Libraries
import numpy as np
from scipy.stats import rankdata


# %%
# Pilot Selection
class pilotSelection:
    def selectStream(self, methodName, budgetPercent, streamScores, triggerThreshold, randomGenerator):
        """
        This method chooses the windows of one stream on which the recognizer runs; the first window always runs and counts toward the budget

        Arguments
        =========
        methodName : Selection method
        budgetPercent : Share of the stream's windows to keep, in percent, rounded up to whole windows
        streamScores : Windows by scores array of the stream
        triggerThreshold : Change threshold of the trigger for this fold and budget
        randomGenerator : Random generator of the random baseline, None for the other methods

        Output
        ======
        Boolean array marking the kept windows
        """
        # Keep The First Window And Work Out How Many Windows The Budget Allows
        windowCount = len(streamScores)
        keepCount = (budgetPercent * windowCount + 99) // 100
        keptWindows = np.zeros(windowCount, dtype=bool)
        keptWindows[0] = True

        # Apply The Rule Of The Method To The Remaining Windows
        if methodName == "changeTrigger":
            keptWindows[1:] = streamScores[1:, self.changeColumn] >= triggerThreshold
        elif methodName == "uniformStride":
            keptWindows[(np.arange(keepCount) * windowCount) // keepCount] = True
        elif methodName == "random":
            keptWindows[1 + randomGenerator.choice(windowCount - 1, keepCount - 1, replace=False)] = True
        elif keepCount > 1:
            laterScores = streamScores[1:]
            if methodName == "meanRank":
                rankingScores = np.mean([rankdata(laterScores[:, scoreColumn]) for scoreColumn in range(laterScores.shape[1])], axis=0)
            else:
                rankingScores = laterScores[:, self.builder.scoreNames.index(methodName)]
            keptWindows[1 + np.argsort(-rankingScores, kind="stable")[:keepCount - 1]] = True
        return keptWindows

    def selectAll(self, methodName, budgetPercent, seedOffset):
        """
        This method selects windows in every stream and gives every dropped window the prediction of the last kept window of its stream

        Arguments
        =========
        methodName : Selection method
        budgetPercent : Budget in percent
        seedOffset : Offset added to the seed for the random baseline, None for the other methods

        Output
        ======
        Tuple of the kept window mask and the final prediction of every window
        """
        # Seed The Random Baseline From The Seed, Its Offset And The Budget
        randomGenerator = None if seedOffset is None else np.random.default_rng([self.randomSeed + seedOffset, budgetPercent])

        # Select And Carry Predictions Forward Stream By Stream
        keptMask = np.zeros(len(self.fullPredictions), dtype=bool)
        finalPredictions = np.zeros_like(self.fullPredictions)
        for streamStart, streamEnd in self.streamRanges:
            triggerThreshold = self.triggerThresholds[self.foldNumbers[streamStart]][budgetPercent]
            keptWindows = self.selectStream(methodName, budgetPercent, self.scoreMatrix[streamStart:streamEnd], triggerThreshold, randomGenerator)
            lastKept = np.maximum.accumulate(np.where(keptWindows, np.arange(len(keptWindows)), 0))
            keptMask[streamStart:streamEnd] = keptWindows
            finalPredictions[streamStart:streamEnd] = self.fullPredictions[streamStart:streamEnd][lastKept]
        return keptMask, finalPredictions
