# This file is responsible for embedding every window with PCA and t-SNE and plotting the map for the smartphone exploration
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from common.csvWriters import writeCsv
from common.plotDefaults import categoricalColours, saveFigure


# %%
# Embedding Sections
class embeddingSections:
    def embedWindows(self):
        """
        This method embeds every window with a fixed PCA step followed by t-SNE and plots it by activity and by subject ( section 4.1 )

        Arguments
        =========
        None

        Output
        ======
        Array of the two dimensional t-SNE coordinates
        """
        # Reduce The Standardised Features To Principal Components
        pcaModel, principalComponents = self.reduceToComponents()

        # Embed The Components With t-SNE
        tsneModel, tsneCoordinates = self.embedComponents(principalComponents)

        # Record The Embedding Settings
        self.recordEmbeddingSettings(pcaModel, tsneModel)

        # Plot The Embedding By Activity And By Subject
        self.plotEmbedding(tsneCoordinates, self.featureFrame["Activity"].to_numpy(), self.activityOrder, "activity", "tsneByActivity.png", 1)
        subjectOrder = sorted(self.featureFrame["subject"].unique().tolist())
        self.plotEmbedding(tsneCoordinates, self.featureFrame["subject"].to_numpy(), subjectOrder, "subject", "tsneBySubject.png", 2)
        print(f"[ SMARTPHONE : TSNE KL DIVERGENCE ] : {tsneModel.kl_divergence_:.4f}")
        return tsneCoordinates

    def reduceToComponents(self):
        """
        This method standardises the features, projects them onto a fixed number of principal components and writes the explained variance

        Arguments
        =========
        None

        Output
        ======
        Tuple of the fitted PCA model and the principal component scores
        """
        # Standardise The Features
        scaledFeatures = StandardScaler().fit_transform(self.featureFrame[self.featureNames].to_numpy())

        # Project Onto A Fixed Number Of Principal Components
        pcaModel = PCA(n_components=self.pcaComponents, svd_solver="full", random_state=self.randomSeed)
        principalComponents = pcaModel.fit_transform(scaledFeatures)
        varianceTable = pd.DataFrame({
            "component": np.arange(1, self.pcaComponents + 1),
            "explainedVarianceRatio": pcaModel.explained_variance_ratio_,
            "cumulativeExplainedVarianceRatio": np.cumsum(pcaModel.explained_variance_ratio_),
        })
        writeCsv(varianceTable, self.resultsDirectory / "pcaExplainedVariance.csv")
        return pcaModel, principalComponents

    def embedComponents(self, principalComponents):
        """
        This method embeds the principal components in two dimensions with t-SNE

        Arguments
        =========
        principalComponents : Principal component scores of every window

        Output
        ======
        Tuple of the fitted t-SNE model and the two dimensional coordinates
        """
        # Fit t-SNE On The Components
        tsneModel = TSNE(
            n_components=2, perplexity=self.tsnePerplexity, init="pca", learning_rate="auto",
            max_iter=self.tsneIterations, random_state=self.randomSeed, n_jobs=self.nJobs,
        )
        tsneCoordinates = tsneModel.fit_transform(principalComponents)
        return tsneModel, tsneCoordinates

    def recordEmbeddingSettings(self, pcaModel, tsneModel):
        """
        This method records the PCA and t-SNE settings and results for runParameters.csv

        Arguments
        =========
        pcaModel : Fitted PCA model
        tsneModel : Fitted t-SNE model

        Output
        ======
        None
        """
        # Store The Embedding Settings
        self.recordParameter("tsne", "standardised", True)
        self.recordParameter("tsne", "pcaComponents", self.pcaComponents)
        self.recordParameter("tsne", "pcaSolver", "full")
        self.recordParameter("tsne", "pcaExplainedVariance", float(np.sum(pcaModel.explained_variance_ratio_)))
        self.recordParameter("tsne", "perplexity", self.tsnePerplexity)
        self.recordParameter("tsne", "init", tsneModel.init)
        self.recordParameter("tsne", "learningRate", tsneModel.learning_rate)
        self.recordParameter("tsne", "learningRateUsed", float(tsneModel.learning_rate_))
        self.recordParameter("tsne", "maxIterations", self.tsneIterations)
        self.recordParameter("tsne", "nIterAttribute", int(tsneModel.n_iter_))
        self.recordParameter("tsne", "klDivergence", float(tsneModel.kl_divergence_))

    def plotEmbedding(self, tsneCoordinates, groupValues, groupOrder, groupTitle, figureName, legendColumns):
        """
        This method draws one t-SNE scatter plot coloured by a grouping column

        Arguments
        =========
        tsneCoordinates : Two dimensional t-SNE coordinates
        groupValues : Group of every window
        groupOrder : Groups in legend order
        groupTitle : Name of the grouping used in the title
        figureName : Png file name inside the figures folder
        legendColumns : Number of legend columns

        Output
        ======
        Path of the saved figure
        """
        # Draw One Colour Per Group
        figure, axis = plt.subplots(figsize=(9, 7))
        groupColours = categoricalColours(len(groupOrder))
        for groupIndex, groupName in enumerate(groupOrder):
            groupMask = groupValues == groupName
            axis.scatter(tsneCoordinates[groupMask, 0], tsneCoordinates[groupMask, 1], s=3, alpha=0.7, color=groupColours[groupIndex], label=str(groupName))
        axis.set_title(f"t-SNE of every window, coloured by {groupTitle}")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(title=groupTitle, markerscale=4, ncol=legendColumns, bbox_to_anchor=(1.02, 1), loc="upper left")
        return saveFigure(figure, self.figuresDirectory / figureName)
