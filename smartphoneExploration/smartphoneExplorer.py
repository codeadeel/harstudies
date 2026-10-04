# This file is responsible for the smartphone exploration study class that joins its section files
# %%
# Importing Libraries
from common.projectPaths import createPartDirectories
from smartphoneExploration.cadenceSections import cadenceSections
from smartphoneExploration.classificationSections import classificationSections
from smartphoneExploration.embeddingSections import embeddingSections
from smartphoneExploration.explorationSections import explorationSections
from smartphoneExploration.harLoader import harLoader
from smartphoneExploration.loadingSections import loadingSections
from smartphoneExploration.participantSections import participantSections
from smartphoneExploration.rawCadenceEstimator import rawCadenceEstimator
from smartphoneExploration.runSetup import runSetup
from smartphoneExploration.sensorSections import sensorSections
from smartphoneExploration.ssaSections import ssaSections
from smartphoneExploration.staircaseSections import staircaseSections


# %%
# Smartphone Explorer, With Its Sections Inherited From Their Files
class smartphoneExplorer(
    runSetup, loadingSections, explorationSections, embeddingSections, classificationSections,
    participantSections, sensorSections, staircaseSections, ssaSections, cadenceSections
):
    def __init__(self, dataRoot, resultsRoot, randomSeed=42, nJobs=4):
        """
        This class initializes the smartphone exploration with its data loader, output folders, seed and thread count

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        resultsRoot : Root directory that receives every part's results
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Thread count for LightGBM, scikit-learn and the native thread pools ( default : 4 )

        Output
        ======
        None
        """
        # Store The Run Settings
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.loader = harLoader(dataRoot)
        self.resultsDirectory, self.figuresDirectory = createPartDirectories(resultsRoot, "smartphoneExploration")

        # Store The Activity Names Used By Single Sections
        self.walkingActivity = "WALKING"
        self.upstairsActivity = "WALKING_UPSTAIRS"
        self.downstairsActivity = "WALKING_DOWNSTAIRS"

        # Store The Analysis Settings
        self.pcaComponents = 50
        self.tsnePerplexity = 30.0
        self.tsneIterations = 1000
        self.classifierSettings = {
            "random_state": randomSeed, "n_jobs": nJobs, "deterministic": True, "force_col_wise": True, "verbose": -1,
        }
        self.identificationTestFraction = 0.25
        self.boutHeldOutFolds = round(1 / self.identificationTestFraction)
        self.ssaParticipant = 1
        self.ssaMeanFeatures = ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z"]
        self.ssaPairIndices = (1, 2)
        self.ssaTrendIndex = 0
        self.pairCorrelationThreshold = 0.5
        self.cadenceEstimator = rawCadenceEstimator(
            self.loader.samplingRate, lowFrequency=0.5, highFrequency=3.0,
            welchSegmentSamples=512, fftLength=8192, harmonicCount=4,
        )

        # Prepare The State Filled While The Sections Run
        self.parameterRows = []
        self.featureFrame = None
        self.featureNames = None
        self.activityOrder = None
        self.bodyAcceleration = None
        self.totalAcceleration = None
        self.boutNumbers = None
        self.durationTable = None
        self.walkingModel = None
        self.ssaTable = None
