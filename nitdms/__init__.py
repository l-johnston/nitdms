"""nitdms"""
from sys import maxsize
from os import getenv
from pathlib import Path
from waveformDT.waveform import WaveformDT
from nitdms.version import __version__

# use tdms.dll if 64-bit Python, Windows and installed
try:
    path = Path(getenv("PROGRAMFILES"))
except TypeError:
    from nitdms.reader import TdmsFile
else:
    path = path.joinpath("National Instruments/Shared/TDMS/tdms.dll")
    if path.exists() and maxsize > 2 ** 32:
        from nitdms.tdsapi import TdmsFile
    else:
        from nitdms.reader import TdmsFile

__all__ = ["TdmsFile", "WaveformDT"]
