# This file is responsible for writing the paper's printed values and statements with their page and sentence
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Paper Reference
class paperReference:
    def buildPaperReference(self):
        """
        This method writes the paper's printed precision and recall values and its qualitative statements with their page, section and sentence

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the paper references
        """
        # Transcribe The Printed Values ( Percent ) And Statements, Checked Against The Paper Text
        paperSource = "printed in the paper text (ulfblanke.de/research/csur/paper.pdf)"
        referenceRows = self.paperValueRows(paperSource) + self.paperStatementRows(paperSource)
        referenceTable = pd.DataFrame(referenceRows, columns=[
            "referenceId", "referenceType", "section", "page", "scheme", "configuration", "classifier", "selectedFeatures",
            "paperPrecision", "paperRecall", "paperSentence", "source",
        ])
        referenceTable["selectedFeatures"] = referenceTable["selectedFeatures"].astype("Int64")
        writeCsv(referenceTable, self.resultsDirectory / "paperReference.csv")
        return referenceTable

    def paperValueRows(self, paperSource):
        """
        This method lists the precision and recall values that the paper prints, with the sentence that holds each of them

        Arguments
        =========
        paperSource : Source text recorded with every row

        Output
        ======
        List of reference rows of the type value
        """
        return [
            ("5.1-pd-handAccelerometer", "value", "5.1", "33:20", "personDependent", "handAccelerometer", "kNearestNeighbour", None, 76.2, 44.2,
             "training and testing on the same person results in 76.2% precision (44.2% recall) using only the accelerometer attached to the right hand", paperSource),
            ("5.1-pd-allSensors", "value", "5.1", "33:20", "personDependent", "allSensors", "kNearestNeighbour", None, 94.1, 62.4,
             "When using all sensors, precision increases to 94.1% (62.4% recall)", paperSource),
            ("5.1-pi-handAccelerometer", "value", "5.1", "33:20", "personIndependent", "handAccelerometer", "kNearestNeighbour", None, 44.7, 21.4,
             "For the person-independent case, results are lower: 44.7% precision and 21.4% recall", paperSource),
            ("5.1-pi-allSensors", "value", "5.1", "33:20", "personIndependent", "allSensors", "kNearestNeighbour", None, 63.0, 39.5,
             "Using data from all sensors attached to three different positions on the body improves the recognition performance to 63% precision (39.5% recall)", paperSource),
            ("5.4-pd-hand", "value", "5.4", "33:22", "personDependent", "hand", "kNearestNeighbour", None, 87.2, 55.1,
             "The best result for individual placement is obtained at the hand at 87.2% precision and 55.1% recall", paperSource),
            ("5.4-pi-upperArm", "value", "5.4", "33:22", "personIndependent", "upperArm", "kNearestNeighbour", None, 30.2, 11.4,
             "The worst performance is obtained at the upper arm (30.2% precision and 11.4% recall)", paperSource),
            ("5.5-pd-accAll", "value", "5.5", "33:23", "personDependent", "accAll", "kNearestNeighbour", None, 90.4, 58.6,
             "The best results for the person-dependent case are achieved by combining all three acceleration sensors (p = 90.4%, r = 58.6%)", paperSource),
            ("5.5-pd-bestGyroscope", "value", "5.5", "33:23", "personDependent", "bestGyroscopeOnly", "kNearestNeighbour", None, 85.2, 46.4,
             "Using gyroscopes only, the best performance is a precision of 85.2% (recall 46.4%)", paperSource),
            ("5.6-pd-svm", "value", "5.6", "33:24", "personDependent", "allSensors", "supportVectorMachine", None, 96.0, 84.8,
             "the best results for all available data are achieved by SVM (p = 96%, r = 84.8%)", paperSource),
            ("5.6-pd-naiveBayes", "value", "5.6", "33:24", "personDependent", "allSensors", "gaussianNaiveBayes", None, 78.2, 69.2,
             "The worst performance is exhibited by naive Bayes (p = 78.2%, r = 69.2%)", paperSource),
            ("5.6-pd-kNearestNeighbour", "value", "5.6", "33:24", "personDependent", "allSensors", "kNearestNeighbour", None, 94.1, 62.4,
             "k-NN suffers from lowest recall (p = 94.1%, r = 62.4%)", paperSource),
            ("5.7-pd-mRMR150", "value", "5.7", "33:25", "personDependent", "allSensors", "kNearestNeighbour", 150, 91.9, 59.7,
             "the best recognition performance is achieved for a feature set size of S = 150 (person-dependent evaluation: precision: 91.9%, recall: 59.7%", paperSource),
            ("5.7-pi-mRMR150", "value", "5.7", "33:25", "personIndependent", "allSensors", "kNearestNeighbour", 150, 60.8, 37.3,
             "person-independent: precision: 60.8%, recall: 37.3%", paperSource),
        ]

    def paperStatementRows(self, paperSource):
        """
        This method lists the qualitative statements of the paper with their page, section and sentence

        Arguments
        =========
        paperSource : Source text recorded with every row

        Output
        ======
        List of reference rows of the type statement
        """
        return [
            ("5.2-statement", "statement", "5.2", "33:21", "both", "bothSensorSets", "kNearestNeighbour", None, None, None,
             "The best performance is achieved by using mean and variance as features. [...] Combining mean and variance with other features (FFT and zero crossings) leads to a small decrease of performance", paperSource),
            ("5.2-rawStatement", "statement", "5.2", "33:22", "both", "bothSensorSets", "kNearestNeighbour", None, None, None,
             "Using the raw signal (i.e., each frame instead of a window) led to higher recall at the cost of precision", paperSource),
            ("5.6-statement", "statement", "5.6", "33:24", "both", "allSensors", "supportVectorMachine", None, None, None,
             "the best results for all available data are achieved by SVM (p = 96%, r = 84.8%)", paperSource),
            ("5.7-statement", "statement", "5.7", "33:25", "both", "allSensors", "kNearestNeighbour", 150, None, None,
             "In both cases, the best recognition performance is achieved for a feature set size of S = 150", paperSource),
            ("5.3-statement", "statement", "5.3", "33:22", "both", "allSensors", "kNearestNeighbour", None, None, None,
             "We can see that precision reaches a maximum Ws = 1s for both the person-dependent and the person-independent case. [...] At the same time, however, increasing Ws leads to a decrease of recall", paperSource),
        ]
