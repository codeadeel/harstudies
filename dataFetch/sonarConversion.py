# This file is responsible for converting the SONAR archive into float32 arrays, one recording at a time
# %%
# Importing Libraries
import shutil
import zipfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.csvWriters import writeCsv
from dataFetch.sonarConverter import sonarConverter


# %%
# SONAR Conversion
class sonarConversion:
    def __init__(self, nJobs=4, memberPrefix="SONAR_ML/"):
        """
        This class initializes the conversion with the worker count and the folder name inside the archive

        Arguments
        =========
        nJobs : Number of worker processes converting recordings ( default : 4 )
        memberPrefix : Folder inside the archive that holds the csv recordings ( default : SONAR_ML/ )

        Output
        ======
        None
        """
        # Store The Conversion Settings
        self.nJobs = nJobs
        self.memberPrefix = memberPrefix

    def convertArchive(self, archivePath, outputDirectory):
        """
        This method converts every csv recording of the archive without extracting the archive

        Arguments
        =========
        archivePath : Path of the SONAR zip archive
        outputDirectory : Directory that receives the arrays and the index tables

        Output
        ======
        Number of converted recordings
        """
        # Prepare The Output Directories
        recordingDirectory = outputDirectory / "recordings"
        shutil.rmtree(recordingDirectory, ignore_errors=True)
        recordingDirectory.mkdir(parents=True)

        # Convert Every Csv Member In Worker Processes
        converter = sonarConverter()
        with zipfile.ZipFile(archivePath) as archive:
            memberNames = sorted(name for name in archive.namelist() if name.startswith(self.memberPrefix) and name.endswith(".csv"))
        memberResults = Parallel(n_jobs=self.nJobs)(
            delayed(converter.convertMember)(archivePath, memberName, recordingDirectory) for memberName in memberNames
        )

        # Write The Label Codes, The Index Tables And The Channel Names
        labelSamples = self.writeLabelArrays(memberResults, recordingDirectory)
        writeCsv(pd.DataFrame([memberResult[0] for memberResult in memberResults]), outputDirectory / "recordingIndex.csv")
        writeCsv(pd.DataFrame({"code": range(len(labelSamples)), "label": list(labelSamples), "samples": list(labelSamples.values())}), outputDirectory / "labelNames.csv")
        writeCsv(pd.DataFrame({
            "channel": converter.channelNames,
            "unit": [converter.channelUnits[channelName.split("_")[1][:-1]] for channelName in converter.channelNames],
        }), outputDirectory / "channelNames.csv")
        sampleCount = int(sum(memberResult[0]["samples"] for memberResult in memberResults))
        print(f"[ DATA FETCH : CONVERTED RECORDINGS ] : {len(memberResults)} recordings, {sampleCount} samples")
        return len(memberResults)

    def writeLabelArrays(self, memberResults, recordingDirectory):
        """
        This method codes the labels with one vocabulary over all recordings, saves the label array of each recording and counts the samples per label

        Arguments
        =========
        memberResults : List of ( index row, label names, label positions ) tuples from the converter
        recordingDirectory : Directory that receives the label arrays

        Output
        ======
        Dictionary from label name to its sample count, in vocabulary order
        """
        # Build The Shared Vocabulary
        labelVocabulary = sorted({labelName for memberResult in memberResults for labelName in memberResult[1]})
        labelSamples = dict.fromkeys(labelVocabulary, 0)

        # Save The Coded Labels Of Every Recording And Count The Samples Per Label
        for indexRow, labelNames, labelPositions in memberResults:
            vocabularyCodes = np.array([labelVocabulary.index(labelName) for labelName in labelNames], dtype=np.uint8)
            np.save(recordingDirectory / f"{indexRow['recording']}_labels.npy", vocabularyCodes[labelPositions])
            for labelName, positionCount in zip(labelNames, np.bincount(labelPositions, minlength=len(labelNames))):
                labelSamples[labelName] += int(positionCount)
        return labelSamples
