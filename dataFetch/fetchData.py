# This file is responsible for downloading the datasets used by the studies, once, into the data folder
# %%
# Importing Libraries
import shutil
import subprocess
import urllib.parse
from pathlib import Path, PurePosixPath

from dataFetch.archiveTools import archiveTools
from dataFetch.sonarConversion import sonarConversion


# %%
# Data Fetcher
class dataFetcher:
    def __init__(self, dataRoot, nJobs=4):
        """
        This class initializes the fetcher with the data folder and the dataset sources

        Arguments
        =========
        dataRoot : Folder that receives every dataset
        nJobs : Number of worker processes converting SONAR recordings ( default : 4 )

        Output
        ======
        None
        """
        # Store The Fetch Settings
        self.dataRoot = Path(dataRoot)
        self.downloadDirectory = self.dataRoot / "downloads"
        self.tools = archiveTools()
        self.nJobs = nJobs

        # Describe The Sources, With The File That Shows A Source Is Already In Place
        self.archiveSources = {
            "uciHar": {
                "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
                "folder": "uciHar", "readyFile": "UCI HAR Dataset/features.txt",
            },
            "mhealth": {
                "url": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
                "folder": "mhealth", "readyFile": "MHEALTHDATASET/mHealth_subject1.log",
            },
        }
        self.repositorySources = {
            "actRecTut": {"url": "https://github.com/andreas-bulling/ActRecTut", "folder": "ActRecTut", "readyFile": "Data/subject1_gesture/data.mat"},
        }
        self.sonarSource = {
            "url": "https://zenodo.org/api/records/7881952/files/SONAR_ML.zip/content", "archiveFile": "SONAR_ML.zip",
            "folder": "sonar", "readyFile": "recordingIndex.csv",
        }

    def fetchArchive(self, sourceName):
        """
        This method downloads one zip archive and extracts it, including a zip archive nested inside it

        Arguments
        =========
        sourceName : Key of the source in the archive source table

        Output
        ======
        Folder that holds the extracted dataset
        """
        # Skip A Dataset That Is Already In Place
        source = self.archiveSources[sourceName]
        extractDirectory = self.dataRoot / source["folder"]
        if (extractDirectory / source["readyFile"]).exists():
            print(f"[ DATA FETCH : ALREADY PRESENT ] : {sourceName}")
            return extractDirectory

        # Download The Archive Unless It Is Already Here
        archivePath = self.downloadDirectory / PurePosixPath(urllib.parse.urlparse(source["url"]).path).name
        if not archivePath.exists():
            print(f"[ DATA FETCH : DOWNLOADING ] : {source['url']}")
            self.tools.downloadArchive(source["url"], archivePath)

        # Extract The Archive And Any Archive Nested Inside It
        print(f"[ DATA FETCH : EXTRACTING ] : {archivePath.name}")
        extractDirectory.mkdir(parents=True, exist_ok=True)
        for nestedArchivePath in self.tools.extractArchive(archivePath, extractDirectory):
            self.tools.extractArchive(nestedArchivePath, nestedArchivePath.parent)
        if not (extractDirectory / source["readyFile"]).exists():
            raise FileNotFoundError(f"{source['readyFile']} missing after extracting {archivePath.name}")
        return extractDirectory

    def fetchRepository(self, sourceName):
        """
        This method makes a shallow clone of one git repository

        Arguments
        =========
        sourceName : Key of the source in the repository source table

        Output
        ======
        Folder that holds the clone
        """
        # Skip A Dataset That Is Already In Place
        source = self.repositorySources[sourceName]
        cloneDirectory = self.dataRoot / source["folder"]
        if (cloneDirectory / source["readyFile"]).exists():
            print(f"[ DATA FETCH : ALREADY PRESENT ] : {sourceName}")
            return cloneDirectory

        # Clone Into A Partial Folder And Rename It After Success
        print(f"[ DATA FETCH : CLONING ] : {source['url']}")
        partialDirectory = cloneDirectory.with_name(cloneDirectory.name + ".partial")
        shutil.rmtree(partialDirectory, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", source["url"], str(partialDirectory)], check=True)
        partialDirectory.rename(cloneDirectory)
        if not (cloneDirectory / source["readyFile"]).exists():
            raise FileNotFoundError(f"{source['readyFile']} missing from the clone of {source['url']}")
        return cloneDirectory

    def fetchSonar(self):
        """
        This method downloads the SONAR archive and converts it into float32 arrays

        Arguments
        =========
        None

        Output
        ======
        Folder that holds the converted recordings
        """
        # Skip A Dataset That Is Already In Place
        source = self.sonarSource
        outputDirectory = self.dataRoot / source["folder"]
        if (outputDirectory / source["readyFile"]).exists():
            print("[ DATA FETCH : ALREADY PRESENT ] : sonar")
            return outputDirectory

        # Download The Archive Unless It Is Already Here, Without Reading Back Its 15 GB Of Members
        archivePath = self.downloadDirectory / source["archiveFile"]
        if not archivePath.exists():
            print(f"[ DATA FETCH : DOWNLOADING ] : {source['url']}")
            self.tools.downloadArchive(source["url"], archivePath, checkMembers=False)

        # Convert Every Recording
        print(f"[ DATA FETCH : CONVERTING ] : {archivePath.name}")
        sonarConversion(nJobs=self.nJobs).convertArchive(archivePath, outputDirectory)
        return outputDirectory

    def fetchAll(self, sourceNames):
        """
        This method fetches the requested datasets in order

        Arguments
        =========
        sourceNames : List of uciHar, mhealth, actRecTut or sonar

        Output
        ======
        None
        """
        # Fetch Every Requested Dataset
        self.downloadDirectory.mkdir(parents=True, exist_ok=True)
        for sourceName in sourceNames:
            if sourceName in self.archiveSources:
                self.fetchArchive(sourceName)
            elif sourceName in self.repositorySources:
                self.fetchRepository(sourceName)
            elif sourceName == "sonar":
                self.fetchSonar()
            else:
                raise ValueError(f"No dataset named {sourceName}")
