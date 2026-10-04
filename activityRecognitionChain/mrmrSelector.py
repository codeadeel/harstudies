# This file is responsible for ranking features with minimum redundancy maximum relevance
# %%
# Importing Libraries
import numpy as np


# %%
# mRMR Feature Selection
class mrmrSelector:
    def __init__(self, maximumFeatures=300):
        """
        This class initializes minimum redundancy maximum relevance selection with the mutual information difference criterion and a fixed three level discretisation

        Arguments
        =========
        maximumFeatures : Length of the ranking that is built ( default : 300 )

        Output
        ======
        None
        """
        # Store The Selection Settings ( Levels Below, Inside And Above The Mean Plus Minus One Standard Deviation )
        self.maximumFeatures = maximumFeatures
        self.levelCount = 3

    def discretiseFeatures(self, trainFeatures):
        """
        This method maps every feature to three levels split at its training mean minus and plus one standard deviation

        Arguments
        =========
        trainFeatures : Training features ( windows , features )

        Output
        ======
        Integer levels ( windows , features )
        """
        # Place Every Value Below, Inside Or Above The Mean Plus Minus One Standard Deviation
        featureMeans = trainFeatures.mean(axis=0)
        featureDeviations = trainFeatures.std(axis=0)
        levelCodes = np.ones(trainFeatures.shape, dtype=np.int64)
        levelCodes[trainFeatures < featureMeans - featureDeviations] = 0
        levelCodes[trainFeatures > featureMeans + featureDeviations] = 2
        return levelCodes

    def mutualInformation(self, jointCounts):
        """
        This method computes mutual information in nats from joint count tables

        Arguments
        =========
        jointCounts : Joint counts ( tables , levels of the first variable , levels of the second variable )

        Output
        ======
        Mutual information of every table
        """
        # Turn Counts Into Joint And Marginal Probabilities
        jointProbabilities = jointCounts / jointCounts.sum(axis=(1, 2), keepdims=True)
        firstMarginals = jointProbabilities.sum(axis=2, keepdims=True)
        secondMarginals = jointProbabilities.sum(axis=1, keepdims=True)
        expectedProbabilities = firstMarginals * secondMarginals

        # Sum p log ( p / ( p1 p2 ) ) Over The Non Empty Cells
        logRatios = np.log(jointProbabilities, out=np.zeros_like(jointProbabilities), where=jointProbabilities > 0)
        logRatios -= np.log(expectedProbabilities, out=np.zeros_like(expectedProbabilities), where=jointProbabilities > 0)
        return (jointProbabilities * logRatios).sum(axis=(1, 2))

    def rankFeatures(self, trainFeatures, trainLabels):
        """
        This method ranks features greedily by relevance minus mean redundancy, ties going to the lower feature index

        Arguments
        =========
        trainFeatures : Training features ( windows , features )
        trainLabels : Training window labels

        Output
        ======
        Feature indices in selection order
        """
        # Encode The Levels And The Classes As Indicator Matrices ( Window Rows, Feature And Level Columns )
        levelCodes = self.discretiseFeatures(trainFeatures)
        windowCount, featureCount = levelCodes.shape
        levelIndicators = (levelCodes[:, :, None] == np.arange(self.levelCount)[None, None, :]).astype(np.float64).reshape(windowCount, -1)
        classIndicators = (trainLabels[:, None] == np.unique(trainLabels)[None, :]).astype(np.float64)

        # Measure The Relevance Of Every Feature To The Class ( Whole Number Counts, So The Matrix Product Is Exact )
        relevance = self.mutualInformation((levelIndicators.T @ classIndicators).reshape(featureCount, self.levelCount, -1))

        # Add The Feature With The Best Relevance Minus Mean Redundancy Each Step
        redundancySums = np.zeros(featureCount)
        isSelected = np.zeros(featureCount, dtype=bool)
        selectionOrder = []
        for stepNumber in range(min(self.maximumFeatures, featureCount)):
            selectionScores = relevance - (redundancySums / stepNumber if stepNumber else 0.0)
            selectionScores[isSelected] = -np.inf
            chosenFeature = int(np.argmax(selectionScores))
            selectionOrder.append(chosenFeature)
            isSelected[chosenFeature] = True

            # Update Every Feature's Redundancy With The Newly Chosen One
            chosenIndicators = levelIndicators[:, chosenFeature * self.levelCount:(chosenFeature + 1) * self.levelCount]
            redundancySums += self.mutualInformation((levelIndicators.T @ chosenIndicators).reshape(featureCount, self.levelCount, self.levelCount))
        return np.asarray(selectionOrder)
