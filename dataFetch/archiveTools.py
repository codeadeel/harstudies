# This file is responsible for downloading zip archives and extracting them safely
# %%
# Importing Libraries
import http.client
import time
import urllib.request
import zipfile
import zlib
from pathlib import PurePosixPath


# %%
# Archive Tools
class archiveTools:
    def __init__(self, downloadAttempts=3, chunkBytes=1048576):
        """
        This class initializes the archive tools with the number of download attempts and the chunk size

        Arguments
        =========
        downloadAttempts : Number of attempts for each download ( default : 3 )
        chunkBytes : Bytes read per chunk while downloading ( default : 1048576 )

        Output
        ======
        None
        """
        # Store The Download Settings
        self.downloadAttempts = downloadAttempts
        self.chunkBytes = chunkBytes

    def downloadArchive(self, url, destinationPath, checkMembers=True):
        """
        This method downloads one zip archive into a partial file and renames it once the transfer is complete

        Arguments
        =========
        url : Address of the archive to download
        destinationPath : Final path of the downloaded archive
        checkMembers : Whether every archive member is read back once, which takes long for very large archives ( default : True )

        Output
        ======
        Path of the downloaded archive
        """
        # Download With Retries
        partialPath = destinationPath.with_name(destinationPath.name + ".part")
        for attemptNumber in range(1, self.downloadAttempts + 1):
            try:
                self.streamToFile(url, partialPath)
                if checkMembers:
                    self.readBackMembers(partialPath)
                partialPath.replace(destinationPath)
                return destinationPath
            except (OSError, http.client.HTTPException, zipfile.BadZipFile) as downloadError:
                # Retry After A Growing Pause
                partialPath.unlink(missing_ok=True)
                print(f"[ DATA FETCH : DOWNLOAD RETRY ] : attempt {attemptNumber} failed for {url} ( {downloadError} )")
                if attemptNumber == self.downloadAttempts:
                    raise
                time.sleep(5 * attemptNumber)

    def streamToFile(self, url, partialPath):
        """
        This method streams one response into a file and checks that all announced bytes arrived

        Arguments
        =========
        url : Address to read
        partialPath : File that receives the bytes

        Output
        ======
        None
        """
        # Stream The Response Into The Partial File
        request = urllib.request.Request(url, headers={"User-Agent": "har-studies-dataFetch"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partialPath, "wb") as fileHandle:
            expectedBytes = response.headers.get("Content-Length")
            for fileChunk in iter(lambda: response.read(self.chunkBytes), b""):
                fileHandle.write(fileChunk)

        # Check The Transfer Is Complete
        receivedBytes = partialPath.stat().st_size
        if expectedBytes is not None and receivedBytes != int(expectedBytes):
            raise OSError(f"received {receivedBytes} of {expectedBytes} bytes")

    def readBackMembers(self, archivePath):
        """
        This method reads every member of a zip archive once and treats any read failure as a damaged archive

        Arguments
        =========
        archivePath : Path of the zip archive

        Output
        ======
        None
        """
        # Read Every Member, Treating Any Failure As A Damaged Archive
        try:
            with zipfile.ZipFile(archivePath) as archive:
                damagedMember = archive.testzip()
        except (zlib.error, EOFError, NotImplementedError, RuntimeError, ValueError) as readError:
            raise zipfile.BadZipFile(f"unreadable archive ( {readError} )") from readError
        if damagedMember is not None:
            raise zipfile.BadZipFile(f"damaged member {damagedMember}")

    def extractArchive(self, archivePath, targetDirectory):
        """
        This method extracts a zip archive while skipping macOS metadata and unsafe member paths

        Arguments
        =========
        archivePath : Path of the zip archive
        targetDirectory : Directory that receives the extracted members

        Output
        ======
        List of zip archives found among the extracted members
        """
        # Extract Every Member Except macOS Metadata
        nestedArchives = []
        targetRoot = targetDirectory.resolve()
        with zipfile.ZipFile(archivePath) as archive:
            for member in archive.infolist():
                memberPath = PurePosixPath(member.filename)
                if "__MACOSX" in memberPath.parts or memberPath.name == ".DS_Store":
                    continue

                # Refuse Members That Would Leave The Target Directory
                if not (targetRoot / memberPath).resolve().is_relative_to(targetRoot):
                    raise ValueError(f"Unsafe archive member {member.filename} in {archivePath.name}")
                archive.extract(member, targetDirectory)

                # Remember Nested Archives
                if not member.is_dir() and memberPath.suffix.lower() == ".zip":
                    nestedArchives.append(targetDirectory / memberPath)
        return nestedArchives
