# This file is responsible for the left-right hidden Markov model classifier with one Gaussian HMM per class
# %%
# Importing Libraries
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from scipy.special import logsumexp


# %%
# Left Right Hidden Markov Model Classifier
class leftRightHmmClassifier:
    def __init__(self, stateCount=3, emIterations=10, stayProbability=0.9, randomSeed=42, nJobs=1):
        """
        This class initializes one left-right Gaussian HMM per class, classified by the highest sequence likelihood

        Arguments
        =========
        stateCount : Number of hidden states ( default : 3 )
        emIterations : Maximum Baum-Welch iterations ( default : 10 )
        stayProbability : Initial self transition probability of every state but the last ( default : 0.9 )
        randomSeed : Seed handed to hmmlearn ( default : 42 )
        nJobs : Number of worker processes that fit the class models in parallel ( default : 1 )

        Output
        ======
        None
        """
        # Store The Model Settings
        self.stateCount = stateCount
        self.emIterations = emIterations
        self.stayProbability = stayProbability
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.classLabels = None
        self.classModels = []

    def initialTransitions(self):
        """
        This method builds the left-right transition matrix: stay or move one state forward, the last state absorbing

        Arguments
        =========
        None

        Output
        ======
        Transition matrix ( states , states )
        """
        # Place Stay And Advance Probabilities On The Diagonal And The First Upper Diagonal
        transitionMatrix = np.diag(np.full(self.stateCount, self.stayProbability))
        transitionMatrix += np.diag(np.full(self.stateCount - 1, 1 - self.stayProbability), k=1)
        transitionMatrix[-1, -1] = 1.0
        return transitionMatrix

    def fitClassModel(self, classSequences):
        """
        This method fits the HMM of one class, starting every state from an equal time split of the class sequences

        Arguments
        =========
        classSequences : Training windows of one class ( windows , frames , channels )

        Output
        ======
        Fitted hmmlearn model
        """
        # Start Each State From The Matching Third Of Every Sequence
        stateFrames = [np.concatenate(framePart) for framePart in zip(*[np.array_split(sequence, self.stateCount) for sequence in classSequences])]
        classModel = GaussianHMM(
            n_components=self.stateCount, covariance_type="diag", n_iter=self.emIterations,
            init_params="", params="tmc", random_state=self.randomSeed,
        )
        classModel.startprob_ = np.eye(self.stateCount)[0]
        classModel.transmat_ = self.initialTransitions()
        classModel.means_ = np.stack([framePart.mean(axis=0) for framePart in stateFrames])
        classModel.covars_ = np.stack([framePart.var(axis=0) + classModel.min_covar for framePart in stateFrames])

        # Refine With Baum-Welch On The Concatenated Sequences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classModel.fit(classSequences.reshape(-1, classSequences.shape[2]), lengths=[classSequences.shape[1]] * len(classSequences))
        return classModel

    def fit(self, trainSequences, trainLabels):
        """
        This method fits one HMM per class, the classes in parallel worker processes

        Arguments
        =========
        trainSequences : Training windows ( windows , frames , channels )
        trainLabels : Training window labels

        Output
        ======
        The fitted classifier
        """
        # Fit The Class Models In Class Order
        self.classLabels = np.unique(trainLabels)
        self.classModels = Parallel(n_jobs=self.nJobs)(
            delayed(self.fitClassModel)(trainSequences[trainLabels == classLabel]) for classLabel in self.classLabels
        )
        return self

    def scoreSequences(self, testSequences):
        """
        This method computes the log likelihood of every sequence under every class model with a vectorised forward pass

        Arguments
        =========
        testSequences : Windows to score ( windows , frames , channels )

        Output
        ======
        Log likelihoods ( windows , classes )
        """
        # Score Every Class Model
        sequenceScores = np.zeros((len(testSequences), len(self.classModels)))
        for modelIndex, classModel in enumerate(self.classModels):
            # Take The Diagonal Gaussian Log Density Of Every Frame In Every State
            stateVariances = classModel.covars_
            if stateVariances.ndim == 3:
                stateVariances = np.diagonal(stateVariances, axis1=1, axis2=2)
            logEmissions = np.stack([
                -0.5 * (np.log(2 * np.pi * stateVariances[stateIndex]) + (testSequences - classModel.means_[stateIndex]) ** 2 / stateVariances[stateIndex]).sum(axis=2)
                for stateIndex in range(self.stateCount)
            ], axis=2)

            # Run The Forward Recursion In Log Space
            with np.errstate(divide="ignore"):
                logStart = np.log(classModel.startprob_)
                logTransitions = np.log(classModel.transmat_)
            logForward = logStart[None, :] + logEmissions[:, 0, :]
            for frameIndex in range(1, testSequences.shape[1]):
                logForward = logsumexp(logForward[:, :, None] + logTransitions[None, :, :], axis=1) + logEmissions[:, frameIndex, :]
            sequenceScores[:, modelIndex] = logsumexp(logForward, axis=1)
        return sequenceScores

    def predict(self, testSequences):
        """
        This method labels every sequence with the class of the most likely model

        Arguments
        =========
        testSequences : Windows to label ( windows , frames , channels )

        Output
        ======
        Predicted label of every window
        """
        # Take The Most Likely Class
        return self.classLabels[np.argmax(self.scoreSequences(testSequences), axis=1)]
