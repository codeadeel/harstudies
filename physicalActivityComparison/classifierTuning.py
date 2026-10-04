# This file is responsible for tuning the neighbour count, the tree count and the support vector machine grid on the inner split of the training rows
# %%
# Importing Libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC


# %%
# Classifier Tuning
class classifierTuning:
    def capRows(self, rowPositions, rowLabels, maximumRows):
        """
        This method keeps exactly the given number of rows, or all rows when there are fewer, drawing every class in proportion to its size

        Arguments
        =========
        rowPositions : Row positions to draw from
        rowLabels : Label of every row position
        maximumRows : Number of rows kept

        Output
        ======
        Sorted kept row positions
        """
        # Keep Every Row When The Set Is Small Enough
        if len(rowPositions) <= maximumRows:
            return rowPositions

        # Share The Rows Out By Largest Remainder, Ties To The Lower Label
        classLabels, classSizes = np.unique(rowLabels, return_counts=True)
        exactQuotas = maximumRows * classSizes / len(rowPositions)
        classQuotas = np.floor(exactQuotas).astype(np.int64)
        remainderOrder = np.argsort(-(exactQuotas - classQuotas), kind="stable")
        classQuotas[remainderOrder[:maximumRows - classQuotas.sum()]] += 1

        # Draw Each Class With The Seed
        randomGenerator = np.random.default_rng(self.randomSeed)
        keptParts = [
            randomGenerator.choice(rowPositions[rowLabels == classLabel], size=classQuota, replace=False)
            for classLabel, classQuota in zip(classLabels, classQuotas)
        ]
        return np.sort(np.concatenate(keptParts))

    def tuneNeighbours(self, trainFeatures, trainLabels, validFeatures, validLabels):
        """
        This method scores every candidate K on the validation rows with a single neighbour query and returns the most accurate K

        Arguments
        =========
        trainFeatures : Inner training features
        trainLabels : Inner training labels
        validFeatures : Validation features
        validLabels : Validation labels

        Output
        ======
        Best K, the smallest on ties
        """
        # Find The Neighbours Once And Vote Over Every Prefix, Ties To The Lowest Label As In scikit-learn
        neighbourIndices = NearestNeighbors(n_neighbors=max(self.neighbourCounts)).fit(trainFeatures).kneighbors(validFeatures, return_distance=False)
        neighbourLabels = np.searchsorted(self.classLabels, trainLabels[neighbourIndices])
        labelVotes = np.zeros((len(validLabels), len(self.classLabels)))
        bestCount, bestAccuracy = self.neighbourCounts[0], -1.0
        for neighbourCount in range(1, max(self.neighbourCounts) + 1):
            np.add.at(labelVotes, (np.arange(len(validLabels)), neighbourLabels[:, neighbourCount - 1]), 1)
            accuracy = np.mean(self.classLabels[np.argmax(labelVotes, axis=1)] == validLabels)
            if neighbourCount in self.neighbourCounts and accuracy > bestAccuracy:
                bestCount, bestAccuracy = neighbourCount, accuracy
        return bestCount

    def tuneTrees(self, trainFeatures, trainLabels, validFeatures, validLabels):
        """
        This method grows the largest candidate forest once and scores the prefix of its trees at every candidate size on the validation rows

        Arguments
        =========
        trainFeatures : Inner training features
        trainLabels : Inner training labels
        validFeatures : Validation features
        validLabels : Validation labels

        Output
        ======
        Best number of trees, the smallest on ties
        """
        # Average The Tree Probabilities Prefix By Prefix
        tuningForest = RandomForestClassifier(n_estimators=max(self.treeCounts), random_state=self.randomSeed, n_jobs=1).fit(trainFeatures, trainLabels)
        summedProbabilities = np.zeros((len(validLabels), len(tuningForest.classes_)))
        bestCount, bestAccuracy = self.treeCounts[0], -1.0
        for treeCount, treeModel in enumerate(tuningForest.estimators_, start=1):
            summedProbabilities += treeModel.predict_proba(validFeatures)
            accuracy = np.mean(tuningForest.classes_[np.argmax(summedProbabilities, axis=1)] == validLabels)
            if treeCount in self.treeCounts and accuracy > bestAccuracy:
                bestCount, bestAccuracy = treeCount, accuracy
        return bestCount

    def tuneSupportVectorMachine(self, trainFeatures, trainLabels, validFeatures, validLabels):
        """
        This method scores every cost and gamma of the base 2 grid on the validation rows

        Arguments
        =========
        trainFeatures : Inner training features
        trainLabels : Inner training labels
        validFeatures : Validation features
        validLabels : Validation labels

        Output
        ======
        Tuple of the best base 2 logarithms of C and gamma, the first in grid order on ties
        """
        # Try The Grid In Ascending Order And Keep Strict Improvements
        bestPair, bestAccuracy = (self.svmLogCosts[0], self.svmLogGammas[0]), -1.0
        for logCost in self.svmLogCosts:
            for logGamma in self.svmLogGammas:
                gridModel = SVC(C=2.0 ** logCost, gamma=2.0 ** logGamma).fit(trainFeatures, trainLabels)
                accuracy = np.mean(gridModel.predict(validFeatures) == validLabels)
                if accuracy > bestAccuracy:
                    bestPair, bestAccuracy = (logCost, logGamma), accuracy
        return bestPair
