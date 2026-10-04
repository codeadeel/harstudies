# This file is responsible for cutting one recording into fixed size sliding windows, labelling them and mapping frames to them
# %%
# Importing Libraries
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# %%
# Sliding Windows
class slidingWindows:
    def __init__(self, frameCount, windowSamples, stepSamples):
        """
        This class initializes fixed size sliding windows over one recording

        Arguments
        =========
        frameCount : Number of frames in the recording
        windowSamples : Window length in frames
        stepSamples : Step between window starts in frames

        Output
        ======
        None
        """
        # Check The Window Settings
        if windowSamples < 1 or stepSamples < 1 or windowSamples > frameCount:
            raise ValueError(f"Invalid window of {windowSamples} frames with step {stepSamples} for {frameCount} frames")

        # Place The Windows
        self.frameCount = frameCount
        self.windowSamples = windowSamples
        self.stepSamples = stepSamples
        self.windowStarts = np.arange(0, frameCount - windowSamples + 1, stepSamples)
        self.centreFrames = self.windowStarts + windowSamples // 2

    def labelWindows(self, frameLabels, classLabels):
        """
        This method gives every window the majority label of its frames, a tie going to the centre frame's label when that label is among the tied ones and otherwise to the lowest tied label

        Arguments
        =========
        frameLabels : Integer label of every frame
        classLabels : Sorted class labels

        Output
        ======
        Label of every window
        """
        # Count Every Class In Every Window With Cumulative Sums
        classLabels = np.asarray(classLabels)
        labelCounts = np.zeros((len(self.windowStarts), len(classLabels)), dtype=np.int64)
        for labelIndex, classLabel in enumerate(classLabels):
            cumulativeCount = np.concatenate([[0], np.cumsum(frameLabels == classLabel)])
            labelCounts[:, labelIndex] = cumulativeCount[self.windowStarts + self.windowSamples] - cumulativeCount[self.windowStarts]

        # Take The Majority, Or The Centre Frame's Label When It Ties For The Majority
        topCounts = labelCounts.max(axis=1)
        majorityLabels = classLabels[np.argmax(labelCounts, axis=1)]
        isTie = (labelCounts == topCounts[:, None]).sum(axis=1) > 1
        centreLabels = frameLabels[self.centreFrames]
        centreCounts = labelCounts[np.arange(len(self.windowStarts)), np.searchsorted(classLabels, centreLabels)]
        return np.where(isTie & (centreCounts == topCounts), centreLabels, majorityLabels)

    def mapFramesToWindows(self, frameIndices, firstWindow, lastWindow):
        """
        This method maps frames to the window whose centre is nearest, ties going to the earlier window, within a window range

        Arguments
        =========
        frameIndices : Frames to map
        firstWindow : First window index allowed
        lastWindow : Last window index allowed

        Output
        ======
        Window index of every frame
        """
        # Solve The Nearest Geometric Centre With Integer Arithmetic ( Window j Spans Frames j * step To j * step + window - 1 )
        numerator = 2 * np.asarray(frameIndices) + 1 - self.windowSamples - self.stepSamples
        nearestWindows = -((-numerator) // (2 * self.stepSamples))
        return np.clip(nearestWindows, firstWindow, lastWindow)

    def windowSpan(self, firstWindow, lastWindow):
        """
        This method returns the frames covered by a range of windows

        Arguments
        =========
        firstWindow : First window index of the range
        lastWindow : Last window index of the range

        Output
        ======
        Tuple of the first covered frame and the frame after the last covered frame
        """
        # Join The First Start And The Last End
        return int(self.windowStarts[firstWindow]), int(self.windowStarts[lastWindow] + self.windowSamples)

    def cutSequences(self, dataMatrix):
        """
        This method returns the frames of every window as one array

        Arguments
        =========
        dataMatrix : Data matrix ( frames , channels )

        Output
        ======
        Array of shape ( windows , window samples , channels )
        """
        # Take Every Window As A View And Copy It Once
        windowView = sliding_window_view(dataMatrix, self.windowSamples, axis=0)
        return np.ascontiguousarray(windowView[self.windowStarts].transpose(0, 2, 1))
