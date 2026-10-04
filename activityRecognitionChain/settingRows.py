# This file is responsible for listing the feature and classifier settings for runParameters.csv
# %%
# Importing Libraries
from activityRecognitionChain.leftRightHmmClassifier import leftRightHmmClassifier


# %%
# Setting Rows
class settingRows:
    def featureSettingRows(self):
        """
        This method lists the settings and definitions of the feature types

        Arguments
        =========
        None

        Output
        ======
        List of ( section , parameter , value ) tuples
        """
        # List The Feature Settings
        extractor = self.evaluator.extractor
        bandFractions = ", ".join(f"1/{2 ** power}" for power in range(extractor.bandCount - 1, 0, -1))
        return [
            ("features", "bandCount", extractor.bandCount),
            ("features", "bandUpperEdges", f"{bandFractions} and all of the spectrum"),
            ("features", "cepstralCount", extractor.cepstralCount),
            ("features", "allFeaturesPerChannel", len(self.evaluator.getFeatures(
                self.evaluator.loader.subjects[0], self.evaluator.secondsToSamples(self.windowSeconds), self.evaluator.secondsToSamples(self.stepSeconds), "All",
            )[1]) // len(self.evaluator.loader.channelNames)),
            ("features", "energyDefinition", "the one-sided power sum of the mean removed window divided by the window length, about half the window length times the variance"),
            ("features", "rawDefinition", "Raw uses every frame as one instance (one frame windows), as the paper defines it; RawWindow concatenates the frames of each paper window"),
        ]

    def classifierSettingRows(self):
        """
        This method lists the settings of the classifiers, the SVM search, the HMM and mRMR

        Arguments
        =========
        None

        Output
        ======
        List of ( section , parameter , value ) tuples
        """
        # List The Classifier Settings
        factory = self.evaluator.factory
        hmmDefaults = leftRightHmmClassifier()
        return [
            ("classifiers", "kNearestNeighbourK", factory.buildClassifier("kNearestNeighbour").n_neighbors),
            ("classifiers", "toolboxNearestNeighboursK", factory.buildClassifier("toolboxNearestNeighbours").n_neighbors),
            ("classifiers", "svmKernel", "rbf"),
            ("classifiers", "svmCostGrid", ", ".join(f"{costValue:g}" for costValue in factory.svmCostGrid)),
            ("classifiers", "svmGammaGrid", ", ".join(f"{gammaValue:g}" for gammaValue in factory.svmGammaGrid)),
            ("classifiers", "svmInnerFolds", factory.innerFolds),
            ("classifiers", "svmSearch", "GroupKFold over the repetitions of the training rows, scored by macro F1 on windows"),
            ("classifiers", "svmGridNote", "the grid was widened after trial runs whose inner search chose the upper edge of narrower grids, first for C and then for gamma"),
            ("classifiers", "adaBoostRounds", factory.boostingRounds),
            ("classifiers", "adaBoostWeakLearner", "depth-one decision trees (the scikit-learn default)"),
            ("classifiers", "toolboxStepClassifiers", ", ".join(self.toolboxStepClassifiers)),
            ("classifiers", "hmmStates", hmmDefaults.stateCount), ("classifiers", "hmmEmIterations", hmmDefaults.emIterations),
            ("classifiers", "hmmTopology", f"left-right topology, start in state one, initial stay probability {hmmDefaults.stayProbability:g}, one diagonal Gaussian per state"),
            ("classifiers", "hmmInput", "raw frames of the window, channels z-scored with the frames the training windows cover"),
            ("classifiers", "hmmDecision", "there is one model per class, and each test window goes to the class whose model gives it the highest log-likelihood"),
            ("classifiers", "mrmrCriterion", "mutual information difference (relevance minus mean redundancy)"),
            ("classifiers", "mrmrDiscretisation", "three levels split at the training mean minus and plus one standard deviation"),
        ]
