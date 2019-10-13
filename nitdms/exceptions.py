"""Exceptions"""


class TdmsException(Exception):
    """Base exception class for this module"""


class InvalidTDMSFileError(TdmsException):
    """Raised when the file does not have the expected 'TDSm' lead-in"""


class InvalidTDMSVersionError(TdmsException):
    """Raised when the lead-in TDMS version doesn't match the expected value"""


class SegmentCorruptedError(TdmsException):
    """Raised when leadin segment length is 0xFFFFFFFFFFFFFFFF"""


class DataTypeNotSupportedError(TdmsException):
    """Raised when TDMS file has a data type that isn't supported"""


class InvalidDimensionError(TdmsException):
    """Raised when raw data index array dimension is > 1"""


class DAQmxScaleTypeError(TdmsException):
    """Raised when DAQmx scale type is not supported"""
