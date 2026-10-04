# This file is responsible for listing the classifier and protocol rows of the parameter table
# %%
# Importing Libraries
from physicalActivityComparison.flooredGmmHmm import flooredGmmHmm


# %%
# Method Parameter Steps
class methodParameterSteps:
    def listClassifierRows(self):
        """
        This method lists the feature selection rule, the tuning rule and the settings of every classifier

        Arguments
        =========
        None

        Output
        ======
        List of tuples: section, parameter and value
        """
        # Name The Inner Split In Words
        classifiers = self.classifiers
        innerOrdinal = f"{self.folds.innerFoldCount}{'th' if 10 <= self.folds.innerFoldCount % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(self.folds.innerFoldCount % 10, 'th')}"

        # Record The Selection And Classifier Settings
        return [
            ("selection", "rule", f"impurity importance of a {classifiers.selectionTrees} tree random forest on the training rows of each fold, keeping the smallest top set whose importance exceeds {100 * classifiers.importanceShare:g}% of the total"),
            ("classifiers", "scaling", "z-score with the training rows of the fold"),
            ("classifiers", "tuningRule", f"one inner split of the training rows that holds out one part in {self.folds.innerFoldCount}, made the way the protocol makes its outer split: one of {self.folds.innerFoldCount} stratified folds for P1 and every {innerOrdinal} remaining participant for P3; the best accuracy there wins, ties going to the smaller setting"),
            ("protocol", "innerFoldCount", self.folds.innerFoldCount),
            ("classifiers", "neighbourCounts", ", ".join(str(neighbourCount) for neighbourCount in classifiers.neighbourCounts)),
            ("classifiers", "treeCounts", ", ".join(str(treeCount) for treeCount in classifiers.treeCounts)),
            ("classifiers", "svmLog2CostGrid", ", ".join(str(logCost) for logCost in classifiers.svmLogCosts)),
            ("classifiers", "svmLog2GammaGrid", ", ".join(str(logGamma) for logGamma in classifiers.svmLogGammas)),
            ("classifiers", "svmTuningRows", f"the grid is fitted on {classifiers.svmTuningRows} inner training rows and scored on {classifiers.svmValidationRows} validation rows (all rows where there are fewer), each class drawn in proportion with the seed, and the chosen C and gamma are refitted on all training rows"),
            ("classifiers", "supervisedLearningGaussianMixture", "one diagonal Gaussian per class, the class proportions of the training rows as priors, and the class of highest posterior probability (scikit-learn GaussianNB)"),
            ("classifiers", "hiddenMarkovModel", f"hmmlearn GMMHMM, {classifiers.clusterCount} states, ergodic, {classifiers.hmmMixtures} diagonal Gaussians per state, {classifiers.hmmIterations} EM iterations"),
            ("classifiers", "hmmIterations", classifiers.hmmIterations),
            ("classifiers", "hmmCovarianceFloor", f"every variance is floored at hmmlearn's min_covar of {flooredGmmHmm(n_components=1).min_covar:g} after each M-step, because discrete features such as zero crossings can collapse a mixture component onto one value"),
            ("classifiers", "hmmSequences", "training rows in time order per participant form the training sequences; the test rows in time order per participant are decoded with Viterbi as their own sequences"),
            ("classifiers", "kMeans", f"{classifiers.clusterCount} clusters, {classifiers.kMeansInitialisations} initialisations"),
            ("classifiers", "gaussianMixture", f"{classifiers.clusterCount} diagonal Gaussians"),
            ("classifiers", "clusterMatching", "Hungarian assignment of clusters or states to labels, maximising the matched training rows of the fold"),
        ]

    def listProtocolRows(self):
        """
        This method lists the two evaluation protocols, the metrics and the way the folds are averaged

        Arguments
        =========
        None

        Output
        ======
        List of tuples: section, parameter and value
        """
        # Record The Protocol Settings
        return [
            ("protocol", "shuffledFolds", f"{self.folds.foldCount} stratified folds over shuffled rows"),
            ("protocol", "leaveOneSubjectOut", "each participant is the test set once"),
            ("protocol", "metrics", f"accuracy; precision, recall and specificity per class averaged over the classes; F as in Attal et al.'s equation 10 with beta {self.fBeta:g} from the averaged precision and recall; macro F1 as the mean of the per class F1"),
            ("protocol", "fBeta", self.fBeta), ("protocol", "paperFTolerance", self.fTolerance),
            ("protocol", "foldAveraging", "each metric is computed per fold and averaged over the folds; standard deviations use one degree of freedom less than the fold count"),
        ]
