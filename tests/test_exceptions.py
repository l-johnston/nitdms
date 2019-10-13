"""Test exceptions"""
from pytest import raises
from nitdms import TdmsFile
from nitdms.exceptions import (
    InvalidTDMSFileError,
    InvalidTDMSVersionError,
    InvalidDimensionError,
    SegmentCorruptedError,
    DataTypeNotSupportedError,
)

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member


def test_invalidfileerror():
    with raises(InvalidTDMSFileError):
        TdmsFile("./tests/tdms_files/invalid file.txt")


def test_invalidversionerror():
    with raises(InvalidTDMSVersionError):
        TdmsFile("./tests/tdms_files/invalid version.tdms")


def test_invaliddimensionerror():
    with raises(InvalidDimensionError):
        TdmsFile("./tests/tdms_files/invalid dimension.tdms")


def test_segmentcorruptederror():
    with raises(SegmentCorruptedError):
        TdmsFile("./tests/tdms_files/segment corrupted.tdms")


def test_datatypenotsupporterror():
    with raises(DataTypeNotSupportedError):
        TdmsFile("./tests/tdms_files/extended data type.tdms")
