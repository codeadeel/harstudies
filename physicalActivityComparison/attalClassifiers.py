# This file is responsible for the settings of the classifiers of Attal et al. ( 2015 ) and the forest ranking that selects the features
# %%
# Importing Libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from physicalActivityComparison.classifierTuning import classifierTuning
from physicalActivityComparison.supervisedFitting import supervisedFitting
from physicalActivityComparison.unsupervisedFitting import unsupervisedFitting


# %%
# Attal Classifiers
class attalClassifiers(classifierTuning, supervisedFitting, unsupervisedFitting):
    def __init__(
        self, classLabels, randomSeed=42, neighbourCounts=(1, 3, 5), treeCounts=(20, 100), svmLogCosts=(1, 2, 6), svmLogGammas=(-7, -5, -3),
        svmTuningRows=3000, svmValidationRows=1500, selectionTrees=50, importanceShare=0.8, kMeansInitialisations=10, hmmMixtures=2, hmmIterations=20,
    ):
        """
        This class initializes the classifier settings: the tuning ranges of Attal et al., the forest selection rule and the unsupervised model sizes

        Arguments
        =========
        classLabels : Activity labels in ascending order
        randomSeed : Seed for every random operation ( default : 42 )
        neighbourCounts : Values of K tried for k-NN ( default : (1, 3, 5) )
        treeCounts : Numbers of trees tried for the random forest ( default : (20, 100) )
        svmLogCosts : Base 2 logarithms of the SVM costs searched ( default : (1, 2, 6) )
        svmLogGammas : Base 2 logarithms of the RBF gammas searched ( default : (-7, -5, -3) )
        svmTuningRows : Number of inner training rows the SVM grid is fitted on, all rows when fewer ( default : 3000 )
        svmValidationRows : Number of validation rows the SVM grid is scored on, all rows when fewer ( default : 1500 )
        selectionTrees : Number of trees of the forest that ranks the features ( default : 50 )
        importanceShare : Share of the total importance the kept features must exceed ( default : 0.8 )
        kMeansInitialisations : Number of k-means initialisations ( default : 10 )
        hmmMixtures : Gaussians per hidden state of the HMM ( default : 2 )
        hmmIterations : EM iterations of the HMM ( default : 20 )

        Output
        ======
        None
        """
        # Store The Tuning Ranges And Model Sizes
        self.classLabels = np.asarray(classLabels)
        self.randomSeed = randomSeed
        self.neighbourCounts = sorted(neighbourCounts)
        self.treeCounts = sorted(treeCounts)
        self.svmLogCosts = list(svmLogCosts)
        self.svmLogGammas = list(svmLogGammas)
        self.svmTuningRows = svmTuningRows
        self.svmValidationRows = svmValidationRows
        self.selectionTrees = selectionTrees
        self.importanceShare = importanceShare
        self.clusterCount = len(classLabels)
        self.kMeansInitialisations = kMeansInitialisations
        self.hmmMixtures = hmmMixtures
        self.hmmIterations = hmmIterations

    def rankFeatures(self, trainFeatures, trainLabels, nJobs=1):
        """
        This method ranks the features by the impurity importance of a random forest and keeps the smallest top set whose importance exceeds the configured share

        Arguments
        =========
        trainFeatures : Training features ( rows , features )
        trainLabels : Training labels
        nJobs : Threads of the forest, 1 inside the fold workers ( default : 1 )

        Output
        ======
        Tuple of the feature order, the importances in that order and the number of kept features
        """
        # Fit The Ranking Forest And Sort The Importances, Ties To The Lower Feature Index
        rankingForest = RandomForestClassifier(n_estimators=self.selectionTrees, random_state=self.randomSeed, n_jobs=nJobs).fit(trainFeatures, trainLabels)
        featureOrder = np.argsort(-rankingForest.feature_importances_, kind="stable")
        sortedImportances = rankingForest.feature_importances_[featureOrder]

        # Keep Features Until Their Summed Importance Exceeds The Share
        keptCount = int(np.searchsorted(np.cumsum(sortedImportances), self.importanceShare, side="right") + 1)
        return featureOrder, sortedImportances, min(keptCount, len(featureOrder))
