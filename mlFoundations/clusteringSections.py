# This file is responsible for clustering a subsample of the windows with three methods and comparing the clusters with the activities
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.csvWriters import writeCsv


# %%
# Clustering Sections
class clusteringSections:
    def runClustering(self):
        """
        This method clusters a stratified subsample of the windows after PCA with k-means, mean shift and hierarchical clustering and compares the clusters with the activities ( chapter Clustering Algorithms )

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per clustering method
        """
        # Draw A Stratified Subsample Of All Windows
        allLabels = self.featureFrame["Activity"].to_numpy()
        sampleRows = np.sort(train_test_split(
            np.arange(len(allLabels)), train_size=self.clusteringRows, stratify=allLabels, random_state=self.randomSeed,
        )[0])
        sampleLabels = allLabels[sampleRows]

        # Standardise And Reduce The Subsample
        sampleStandard = StandardScaler().fit_transform(self.featureFrame[self.featureNames].to_numpy()[sampleRows])
        componentModel = PCA(n_components=self.clusteringComponents, random_state=self.randomSeed)
        sampleComponents = componentModel.fit_transform(sampleStandard)
        self.recordParameter("clustering", "rows", len(sampleRows))
        self.recordParameter("clustering", "components", self.clusteringComponents)
        self.recordParameter("clustering", "componentVarianceShare", float(componentModel.explained_variance_ratio_.sum()))

        # Cluster With The Three Methods
        meanShiftBandwidth = estimate_bandwidth(sampleComponents, quantile=self.bandwidthQuantile, random_state=self.randomSeed)
        clusterModels = {
            "kMeans": (KMeans(n_clusters=self.clusterCount, n_init=self.kMeansInitialisations, random_state=self.randomSeed), f"{self.clusterCount} clusters, {self.kMeansInitialisations} initialisations"),
            "meanShift": (MeanShift(bandwidth=meanShiftBandwidth, bin_seeding=True, n_jobs=self.nJobs), f"bandwidth {meanShiftBandwidth:.3f} from quantile {self.bandwidthQuantile:g}"),
            "hierarchical": (AgglomerativeClustering(n_clusters=self.clusterCount, linkage="ward"), f"{self.clusterCount} clusters, ward linkage"),
        }
        clusteringRows = []
        for methodName, (clusterModel, methodSetting) in clusterModels.items():
            clusterLabels = clusterModel.fit_predict(sampleComponents)
            clusteringRows.append({
                "method": methodName, "setting": methodSetting, "clusters": len(np.unique(clusterLabels)),
                "adjustedRandIndex": adjusted_rand_score(sampleLabels, clusterLabels),
            })
        clusteringTable = pd.DataFrame(clusteringRows)
        writeCsv(clusteringTable, self.resultsDirectory / "clustering.csv")
        return clusteringTable
