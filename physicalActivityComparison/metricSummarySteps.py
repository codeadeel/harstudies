# This file is responsible for putting the fold metrics in a fixed order and averaging them per variant, protocol and classifier
# %%
# Importing Libraries
from common.csvWriters import writeCsv


# %%
# Metric Summary Steps
class metricSummarySteps:
    def sortRecords(self, metricTable):
        """
        This method puts the fold rows in a fixed order: variant, protocol, classifier and fold

        Arguments
        =========
        metricTable : Fold metric rows in any order

        Output
        ======
        Dataframe of the same rows in the fixed order
        """
        # Sort By The Position Of Every Name In Its Fixed List
        sortOrders = {
            "variant": ["rawSubsample", "rawFull", "features80"],
            "protocol": self.folds.protocolNames,
            "classifier": self.supervisedNames + self.unsupervisedNames,
        }
        return metricTable.sort_values(
            ["variant", "protocol", "classifier", "fold"], kind="stable",
            key=lambda sortColumn: sortColumn.map({sortValue: sortIndex for sortIndex, sortValue in enumerate(sortOrders[sortColumn.name])}) if sortColumn.name in sortOrders else sortColumn,
        ).reset_index(drop=True)

    def summariseMetrics(self, metricTable):
        """
        This method averages the fold metrics per variant, protocol and classifier, with the sample standard deviation of the accuracy

        Arguments
        =========
        metricTable : Fold metric table

        Output
        ======
        Dataframe with one row per variant, protocol and classifier
        """
        # Average Over Folds, Standard Deviation With One Degree Of Freedom Removed
        groupColumns = ["variant", "protocol", "learning", "classifier"]
        summaryTable = metricTable.groupby(groupColumns, sort=False).agg(
            folds=("fold", "size"), accuracy=("accuracy", "mean"), accuracyStd=("accuracy", lambda foldValues: foldValues.std(ddof=1)),
            attalF=("attalF", "mean"), macroRecall=("macroRecall", "mean"), macroPrecision=("macroPrecision", "mean"),
            macroSpecificity=("macroSpecificity", "mean"), macroF1=("macroF1", "mean"), macroF1Std=("macroF1", lambda foldValues: foldValues.std(ddof=1)),
        ).reset_index()
        writeCsv(summaryTable, self.resultsDirectory / "metricSummary.csv")
        return summaryTable
