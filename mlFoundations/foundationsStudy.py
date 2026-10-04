# This file is responsible for the study class of the tutorialspoint Machine Learning with Python chapters on the UCI HAR features
# %%
# Importing Libraries
from common.plotDefaults import applyPlotDefaults
from common.projectPaths import createPartDirectories
from common.randomSeeds import limitThreads, seedEverything
from mlFoundations.classifierSections import classifierSections
from mlFoundations.clusteringSections import clusteringSections
from mlFoundations.improvementSections import improvementSections
from mlFoundations.loadingSections import loadingSections
from mlFoundations.metricsSections import metricsSections
from mlFoundations.modelHelpers import modelHelpers
from mlFoundations.parameterSections import parameterSections
from mlFoundations.preparationSections import preparationSections
from mlFoundations.regressionSections import regressionSections
from mlFoundations.selectionSections import selectionSections
from mlFoundations.statisticsPlots import statisticsPlots
from mlFoundations.statisticsSections import statisticsSections
from mlFoundations.workflowSections import workflowSections
from smartphoneExploration.harLoader import harLoader


# %%
# Foundations Study, With The Chapters Inherited From The Section Classes
class foundationsStudy(
    modelHelpers, parameterSections, loadingSections, statisticsSections, statisticsPlots, preparationSections, selectionSections,
    classifierSections, regressionSections, clusteringSections, metricsSections, workflowSections, improvementSections,
):
    def __init__(self, dataRoot, resultsRoot, randomSeed=42, nJobs=4):
        """
        This class initializes the tutorial walk through with the UCI HAR loader, the output folders, the seed and the chapter settings

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        resultsRoot : Root directory that receives every part's results
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Thread and worker count for scikit-learn and the native thread pools ( default : 4 )

        Output
        ======
        None
        """
        # Store The Run Settings
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.loader = harLoader(dataRoot)
        self.resultsDirectory, self.figuresDirectory = createPartDirectories(resultsRoot, "mlFoundations")

        # Store The Statistics And Plot Settings
        self.histogramFeatures = ["tBodyAcc-std()-X", "tGravityAcc-mean()-X", "tBodyAccMag-mean()", "tBodyGyroMag-mean()", "fBodyAccMag-mean()", "angle(X,gravityMean)"]
        self.boxFeature = "tBodyAccMag-mean()"
        self.heatmapFeatures = [
            "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z", "tBodyAcc-std()-X", "tBodyAcc-std()-Y", "tBodyAcc-std()-Z",
            "tGravityAcc-mean()-X", "tGravityAcc-mean()-Y", "tGravityAcc-mean()-Z", "tBodyAccMag-mean()", "tBodyGyroMag-mean()", "fBodyAccMag-mean()",
        ]
        self.histogramBins = 50
        self.skewThreshold = 1.0
        self.correlationThreshold = 0.9

        # Store The Preparation And Selection Settings
        self.binarizerThreshold = 0.0
        self.selectedCount = 20
        self.eliminationStep = 0.1

        # Store The Model Settings
        self.logisticIterations = 2000
        self.forestTrees = 100
        self.neighbourCount = 5
        self.ensembleTrees = 100
        self.boostingRounds = 50
        self.regressionTestShare = 0.25
        self.clusteringRows = 2000
        self.clusteringComponents = 10
        self.clusterCount = 6
        self.kMeansInitialisations = 10
        self.bandwidthQuantile = 0.2
        self.groupFolds = 5
        self.gridCosts = [1.0, 10.0, 100.0]
        self.gridGammas = [0.0001, 0.001, 0.01]

        # Prepare The State Filled While The Chapters Run
        self.parameterRows = []
        self.featureFrame = None
        self.featureNames = None
        self.activityOrder = None
        self.trainFeatures = None
        self.testFeatures = None
        self.trainLabels = None
        self.testLabels = None
        self.trainSubjects = None
        self.trainStandard = None
        self.testStandard = None
        self.fittedClassifiers = {}

    def prepareRun(self):
        """
        This method fixes the seeds, limits the thread pools and applies the plot style before the first chapter runs

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
