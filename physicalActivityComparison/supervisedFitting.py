# This file is responsible for tuning and fitting the four supervised classifiers on the training rows of a fold
# %%
# Importing Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# %%
# Supervised Fitting
class supervisedFitting:
    def fitSupervised(self, classifierName, trainFeatures, trainLabels, validSplit):
        """
        This method tunes one supervised classifier on the inner split of the training rows and refits it on all training rows

        Arguments
        =========
        classifierName : kNearestNeighbour, randomForest, supportVectorMachine or supervisedLearningGaussianMixture
        trainFeatures : Scaled training features
        trainLabels : Training labels
        validSplit : Tuple of the inner training and validation row positions inside the training rows

        Output
        ======
        Tuple of the fitted classifier and a dictionary of the chosen settings
        """
        # Tune On The Inner Split Where The Classifier Has Settings
        innerTrain, innerValid = validSplit
        tuningArguments = (trainFeatures[innerTrain], trainLabels[innerTrain], trainFeatures[innerValid], trainLabels[innerValid])
        if classifierName == "kNearestNeighbour":
            neighbourCount = self.tuneNeighbours(*tuningArguments)
            return KNeighborsClassifier(n_neighbors=neighbourCount).fit(trainFeatures, trainLabels), {"neighbours": neighbourCount}
        if classifierName == "randomForest":
            treeCount = self.tuneTrees(*tuningArguments)
            return RandomForestClassifier(n_estimators=treeCount, random_state=self.randomSeed, n_jobs=1).fit(trainFeatures, trainLabels), {"trees": treeCount}
        if classifierName == "supportVectorMachine":
            cappedTrain = self.capRows(innerTrain, trainLabels[innerTrain], self.svmTuningRows)
            cappedValid = self.capRows(innerValid, trainLabels[innerValid], self.svmValidationRows)
            logCost, logGamma = self.tuneSupportVectorMachine(
                trainFeatures[cappedTrain], trainLabels[cappedTrain], trainFeatures[cappedValid], trainLabels[cappedValid],
            )
            return SVC(C=2.0 ** logCost, gamma=2.0 ** logGamma).fit(trainFeatures, trainLabels), {"log2Cost": logCost, "log2Gamma": logGamma}

        # Fit One Diagonal Gaussian Per Class With The Class Proportions As Priors And Take The Highest Posterior
        if classifierName == "supervisedLearningGaussianMixture":
            return GaussianNB().fit(trainFeatures, trainLabels), {}
        raise ValueError(f"No supervised classifier named {classifierName}")
