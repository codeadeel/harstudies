# This file is responsible for splitting a series into singular spectrum components and measuring how well two components separate
# %%
# Importing Libraries
import numpy as np


# %%
# Singular Spectrum Decomposition
class ssaDecomposition:
    def decomposeSeries(self, timeSeries):
        """
        This method splits a series into its elementary reconstructed components

        Arguments
        =========
        timeSeries : One dimensional series to decompose

        Output
        ======
        Tuple of the reconstructed components ( one row per component ) and the singular values
        """
        # Build The Trajectory Matrix
        seriesValues = np.asarray(timeSeries, dtype=np.float64)
        lagCount = len(seriesValues) - self.windowLength + 1
        trajectoryMatrix = np.column_stack(
            [seriesValues[lagIndex:lagIndex + self.windowLength] for lagIndex in range(lagCount)]
        )

        # Decompose The Trajectory Matrix
        leftVectors, singularValues, rightVectors = np.linalg.svd(trajectoryMatrix, full_matrices=False)

        # Turn Each Elementary Matrix Back Into A Series
        reconstructedComponents = np.array([
            self.averageAntiDiagonals(
                singularValues[componentIndex] * np.outer(leftVectors[:, componentIndex], rightVectors[componentIndex])
            )
            for componentIndex in range(len(singularValues))
        ])
        return reconstructedComponents, singularValues

    def averageAntiDiagonals(self, elementaryMatrix):
        """
        This method averages a matrix along its anti diagonals ( hankelisation )

        Arguments
        =========
        elementaryMatrix : Matrix of shape ( window length , lag count )

        Output
        ======
        Series whose value k is the mean of all entries with row index plus column index equal to k
        """
        # Flip The Rows So Anti Diagonals Become Diagonals
        flippedMatrix = elementaryMatrix[::-1]
        rowCount, columnCount = elementaryMatrix.shape
        return np.array([flippedMatrix.diagonal(offset).mean() for offset in range(-rowCount + 1, columnCount)])

    def weightedCorrelation(self, firstComponent, secondComponent):
        """
        This method measures how strongly two reconstructed components mix with the SSA w-correlation

        Arguments
        =========
        firstComponent : First reconstructed component
        secondComponent : Second reconstructed component of the same length

        Output
        ======
        Absolute w-correlation, near one when the two components cannot be separated from each other
        """
        # Weight Each Series Position By How Often It Appears In The Trajectory Matrix
        seriesLength = len(firstComponent)
        lagCount = seriesLength - self.windowLength + 1
        shorterSide = min(self.windowLength, lagCount)
        positionWeights = np.minimum.reduce([
            np.arange(1, seriesLength + 1), np.full(seriesLength, shorterSide), np.arange(seriesLength, 0, -1),
        ])

        # Correlate The Components Under Those Weights
        weightedProduct = np.sum(positionWeights * firstComponent * secondComponent)
        firstNorm = np.sqrt(np.sum(positionWeights * firstComponent ** 2))
        secondNorm = np.sqrt(np.sum(positionWeights * secondComponent ** 2))
        return abs(weightedProduct) / (firstNorm * secondNorm)
