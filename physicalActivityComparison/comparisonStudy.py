# This file is responsible for holding the shared state of the physical activity comparison and joining its step files
# %%
# Importing Libraries
from common.plotDefaults import applyPlotDefaults
from common.projectPaths import createPartDirectories
from common.randomSeeds import limitThreads, seedEverything
from physicalActivityComparison.attalClassifiers import attalClassifiers
from physicalActivityComparison.attalFeatures import attalFeatureExtractor
from physicalActivityComparison.confusionSteps import confusionSteps
from physicalActivityComparison.dataPreparationSteps import dataPreparationSteps
from physicalActivityComparison.foldEvaluationSteps import foldEvaluationSteps
from physicalActivityComparison.foldScoringSteps import foldScoringSteps
from physicalActivityComparison.inventorySteps import inventorySteps
from physicalActivityComparison.methodParameterSteps import methodParameterSteps
from physicalActivityComparison.metricSummarySteps import metricSummarySteps
from physicalActivityComparison.mhealthLoader import mhealthLoader
from physicalActivityComparison.paperComparisonSteps import paperComparisonSteps
from physicalActivityComparison.parameterSteps import parameterSteps
from physicalActivityComparison.protocolComparisonSteps import protocolComparisonSteps
from physicalActivityComparison.protocolFolds import protocolFolds


# %%
# Comparison Study
class comparisonStudy(
    dataPreparationSteps, inventorySteps, foldEvaluationSteps, foldScoringSteps, metricSummarySteps, paperComparisonSteps,
    protocolComparisonSteps, confusionSteps, parameterSteps, methodParameterSteps,
):
    def __init__(self, dataRoot, resultsRoot, randomSeed=42, nJobs=4, rawSampleFraction=0.1):
        """
        This class initializes the study with the loader, the feature extractor, the classifiers, the protocols and the output folders

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        resultsRoot : Root directory that receives every part's results
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Worker processes over the folds and native thread count ( default : 4 )
        rawSampleFraction : Share of the labelled samples kept per participant and class for the supervised raw data case ( default : 0.1 )

        Output
        ======
        None
        """
        # Store The Run Settings And Building Blocks
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.rawSampleFraction = rawSampleFraction
        self.loader = mhealthLoader(dataRoot)
        self.extractor = attalFeatureExtractor(self.loader.signalNames, list(self.loader.placementColumns))
        self.classifiers = attalClassifiers(self.loader.classLabels, randomSeed=randomSeed)
        self.folds = protocolFolds(randomSeed=randomSeed)
        self.resultsDirectory, self.figuresDirectory = createPartDirectories(resultsRoot, "physicalActivityComparison")

        # Store The Classifiers, The Window Steps And The Evaluated Variants
        self.supervisedNames = ["kNearestNeighbour", "randomForest", "supportVectorMachine", "supervisedLearningGaussianMixture"]
        self.unsupervisedNames = ["hiddenMarkovModel", "kMeans", "gaussianMixture"]
        self.windowSteps = {"features80": 5}

        # Store The F-Measure Weight Of Attal Et Al.'s Equation 10 And The Tolerance For Checking Their Printed Values
        self.fBeta = 1.0
        self.fTolerance = 0.01

    def setUpRun(self):
        """
        This method fixes the random seeds, the native thread limits and the plot style before the first step

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Fix The Seeds, Thread Pools And Plot Style
        seedEverything(self.randomSeed)
        limitThreads(self.nJobs)
        applyPlotDefaults()
