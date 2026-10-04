# This file is responsible for fitting the unsupervised models without labels and mapping their clusters to activity labels
# %%
# Importing Libraries
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from physicalActivityComparison.flooredGmmHmm import flooredGmmHmm


# %%
# Unsupervised Fitting
class unsupervisedFitting:
    def matchClusters(self, trainClusters, trainLabels):
        """
        This method maps every cluster or state to one label with the Hungarian algorithm on the training rows

        Arguments
        =========
        trainClusters : Cluster or state number of every training row
        trainLabels : Label of every training row

        Output
        ======
        Array giving the label of every cluster number
        """
        # Count Clusters Against Labels And Maximise The Matched Rows
        contingency = np.zeros((self.clusterCount, len(self.classLabels)))
        np.add.at(contingency, (trainClusters, np.searchsorted(self.classLabels, trainLabels)), 1)
        clusterRows, labelColumns = linear_sum_assignment(-contingency)
        clusterLabels = np.zeros(self.clusterCount, dtype=self.classLabels.dtype)
        clusterLabels[clusterRows] = self.classLabels[labelColumns]
        return clusterLabels

    def fitUnsupervised(self, modelName, trainFeatures, trainLengths, testFeatures, testLengths):
        """
        This method fits one unsupervised model without labels and returns the cluster or state of every training and test row

        Arguments
        =========
        modelName : kMeans, gaussianMixture or hiddenMarkovModel
        trainFeatures : Scaled training rows in time order per participant
        trainLengths : Number of training rows per participant, in the same order
        testFeatures : Scaled test rows in time order per participant
        testLengths : Number of test rows per participant, in the same order

        Output
        ======
        Tuple of the training clusters, the test clusters and a dictionary of fit details
        """
        # Cluster The Rows Without Any Time Order
        if modelName == "kMeans":
            clusterModel = KMeans(n_clusters=self.clusterCount, n_init=self.kMeansInitialisations, random_state=self.randomSeed).fit(trainFeatures)
            return clusterModel.labels_, clusterModel.predict(testFeatures), {}
        if modelName == "gaussianMixture":
            mixtureModel = GaussianMixture(n_components=self.clusterCount, covariance_type="diag", random_state=self.randomSeed).fit(trainFeatures)
            return mixtureModel.predict(trainFeatures), mixtureModel.predict(testFeatures), {"converged": bool(mixtureModel.converged_)}

        # Fit An Ergodic HMM On The Time Ordered Sequences And Decode Each Part With Viterbi
        if modelName == "hiddenMarkovModel":
            stateModel = flooredGmmHmm(
                n_components=self.clusterCount, n_mix=self.hmmMixtures, covariance_type="diag", n_iter=self.hmmIterations, random_state=self.randomSeed,
            ).fit(trainFeatures, trainLengths)
            likelihoodHistory = list(stateModel.monitor_.history)
            fitDetails = {
                "emIterations": stateModel.monitor_.iter,
                "emConverged": bool(len(likelihoodHistory) >= 2 and likelihoodHistory[-1] - likelihoodHistory[-2] < stateModel.monitor_.tol),
            }
            return stateModel.predict(trainFeatures, trainLengths), stateModel.predict(testFeatures, testLengths), fitDetails
        raise ValueError(f"No unsupervised model named {modelName}")
