# This file is responsible for scoring the predictions of one round frame by frame and event by event
# %%
# Importing Libraries
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix


# %%
# Round Scoring
class roundScoring:
    def scoreFrames(self, trueLabels, predictedLabels):
        """
        This method computes per class precision and recall from a frame or window comparison and averages them over classes

        Arguments
        =========
        trueLabels : Ground truth labels
        predictedLabels : Predicted labels

        Output
        ======
        Tuple of the metric dictionary and the confusion matrix ( true classes as rows )
        """
        # Count Hits, Predictions And Ground Truth Per Class
        confusionCounts = confusion_matrix(trueLabels, predictedLabels, labels=self.classLabels)
        truePositives = np.diag(confusionCounts).astype(np.float64)
        predictedCounts = confusionCounts.sum(axis=0)
        trueCounts = confusionCounts.sum(axis=1)

        # Precision Of A Never Predicted Class Is Zero, Or Undefined In The Alternative Average
        precisionZero = np.divide(truePositives, predictedCounts, out=np.zeros_like(truePositives), where=predictedCounts > 0)
        precisionDefined = np.divide(truePositives, predictedCounts, out=np.full_like(truePositives, np.nan), where=predictedCounts > 0)
        classRecall = np.divide(truePositives, trueCounts, out=np.full_like(truePositives, np.nan), where=trueCounts > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            roundMetrics = {
                "precisionAll": float(precisionZero.mean()),
                "recallAll": float(np.nanmean(classRecall)),
                "precisionNonNull": float(precisionZero[self.gestureMask].mean()),
                "recallNonNull": float(np.nanmean(classRecall[self.gestureMask])),
                "precisionAllDefined": float(np.nanmean(precisionDefined)),
                "precisionNonNullDefined": float(np.nanmean(precisionDefined[self.gestureMask])),
                "classesNeverPredicted": int(np.sum(predictedCounts == 0)),
                "classesAbsent": int(np.sum(trueCounts == 0)),
            }
        return roundMetrics, confusionCounts

    def scoreEvents(self, testSubject, testFrames, centreFrames, predictedWindowLabels):
        """
        This method scores windows by the ground truth segment around their centre and counts gesture segments hit by at least one correct window

        Arguments
        =========
        testSubject : Participant number of the test data
        testFrames : Frames of the round
        centreFrames : Centre frame of every test window
        predictedWindowLabels : Predicted label of every test window

        Output
        ======
        Dictionary with the scoreFrames metrics of the windows against their centre labels ( prefixed event ), the number of gesture segments in the round and the number hit by at least one correct window
        """
        # Score Windows Against The Label Of The Segment Holding Their Centre
        frameLabels = self.recordings[testSubject]["labels"]
        windowMetrics, _ = self.scoreFrames(frameLabels[centreFrames], predictedWindowLabels)
        eventMetrics = {f"event{metricName[0].upper()}{metricName[1:]}": metricValue for metricName, metricValue in windowMetrics.items()}

        # Count Gesture Segments Of The Round With At Least One Correct Window Centred Inside
        segmentStarts, segmentEnds, segmentLabels = self.loader.findSegments(frameLabels)
        inRound = (segmentStarts >= testFrames[0]) & (segmentEnds <= testFrames[-1] + 1) & (segmentLabels != self.loader.nullLabel)
        centreSegments = np.searchsorted(segmentStarts, centreFrames, side="right") - 1
        correctSegments = centreSegments[predictedWindowLabels == segmentLabels[centreSegments]]
        segmentHit = np.bincount(correctSegments, minlength=len(segmentStarts)) > 0
        eventMetrics["gestureSegments"] = int(inRound.sum())
        eventMetrics["detectedGestureSegments"] = int((segmentHit & inRound).sum())
        return eventMetrics
