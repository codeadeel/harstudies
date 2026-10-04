# This file is responsible for building the window classifiers and fitting the SVM with its grouped parameter search
# %%
# Importing Libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC


# %%
# Classifier Factory
class classifierFactory:
    def __init__(self, randomSeed=42, nJobs=4, svmCostGrid=(1.0, 10.0, 100.0, 1000.0), svmGammaGrid=(0.001, 0.01, 0.1, 1.0), innerFolds=4, boostingRounds=50, toolboxNeighbourCount=5):
        """
        This class initializes the factory of the window classifiers with fixed seeds and the SVM search grid

        Arguments
        =========
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Worker count of the SVM grid search and the k-NN queries ( default : 4 )
        svmCostGrid : Values of the SVM cost C searched on the training rounds ( default : (1.0, 10.0, 100.0, 1000.0) )
        svmGammaGrid : Values of the RBF gamma searched on the training rounds ( default : (0.001, 0.01, 0.1, 1.0) )
        innerFolds : Number of inner group folds of the SVM grid search ( default : 4 )
        boostingRounds : Number of AdaBoost rounds ( default : 50 )
        toolboxNeighbourCount : Neighbour count of the toolbox's k-NN when no k is passed, used for a labelled sensitivity row ( default : 5 )

        Output
        ======
        None
        """
        # Store The Classifier Settings
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.svmCostGrid = list(svmCostGrid)
        self.svmGammaGrid = list(svmGammaGrid)
        self.innerFolds = innerFolds
        self.boostingRounds = boostingRounds
        self.toolboxNeighbourCount = toolboxNeighbourCount

    def buildClassifier(self, classifierName):
        """
        This method creates one unfitted window classifier

        Arguments
        =========
        classifierName : One of kNearestNeighbour, toolboxNearestNeighbours, nearestClassCentroid, linearDiscriminant, gaussianNaiveBayes and adaBoost ( the SVM and the HMM are fitted by their own methods )

        Output
        ======
        Unfitted scikit-learn classifier
        """
        # Create The Requested Classifier
        if classifierName == "kNearestNeighbour":
            return KNeighborsClassifier(n_neighbors=1, n_jobs=self.nJobs)
        if classifierName == "toolboxNearestNeighbours":
            return KNeighborsClassifier(n_neighbors=self.toolboxNeighbourCount, n_jobs=self.nJobs)
        if classifierName == "nearestClassCentroid":
            return NearestCentroid()
        if classifierName == "linearDiscriminant":
            return LinearDiscriminantAnalysis()
        if classifierName == "gaussianNaiveBayes":
            return GaussianNB()
        if classifierName == "adaBoost":
            return AdaBoostClassifier(n_estimators=self.boostingRounds, random_state=self.randomSeed)
        raise ValueError(f"No plain classifier named {classifierName}")

    def fitSupportVectorMachine(self, trainFeatures, trainLabels, trainGroups):
        """
        This method fits an RBF SVM whose C and gamma are chosen by grouped cross validation inside the training rows only

        Arguments
        =========
        trainFeatures : Scaled training features
        trainLabels : Training window labels
        trainGroups : Repetition block of every training window, the groups of the inner folds

        Output
        ======
        Tuple of the refitted SVM and the chosen parameters
        """
        # Search The Grid On Inner Folds That Keep Repetitions Together
        gridSearch = GridSearchCV(
            SVC(kernel="rbf"), {"C": self.svmCostGrid, "gamma": self.svmGammaGrid},
            scoring="f1_macro", cv=GroupKFold(n_splits=self.innerFolds), n_jobs=self.nJobs, refit=True,
        )
        gridSearch.fit(trainFeatures, trainLabels, groups=trainGroups)
        return gridSearch.best_estimator_, {"svmCost": gridSearch.best_params_["C"], "svmGamma": gridSearch.best_params_["gamma"]}
