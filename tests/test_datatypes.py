"""test data type decoding"""
from datetime import datetime, timezone
from nitdms import TdmsFile

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member

TF = TdmsFile("./tests/tdms_files/data types.tdms")


def test_u8():
    value = TF.U8
    assert isinstance(value, int)
    assert value == 1


def test_i8():
    value = TF.I8
    assert isinstance(value, int)
    assert value == -1


def test_u16():
    value = TF.U16
    assert isinstance(value, int)
    assert value == 1


def test_i16():
    value = TF.I16
    assert isinstance(value, int)
    assert value == -1


def test_u32():
    value = TF.U32
    assert isinstance(value, int)
    assert value == 1


def test_i32():
    value = TF.I32
    assert isinstance(value, int)
    assert value == -1


def test_u64():
    value = TF.U64
    assert isinstance(value, int)
    assert value == 1


def test_i64():
    value = TF.I64
    assert isinstance(value, int)
    assert value == -1


def test_single():
    value = TF.Single
    assert isinstance(value, float)
    assert value == 1.0


def test_double():
    value = TF.Double
    assert isinstance(value, float)
    assert value == -1.0


def test_string():
    value = TF.String
    assert isinstance(value, str)
    assert value == "1"


def test_boolean():
    value = TF.Boolean
    assert isinstance(value, bool)
    assert value


def test_timestamp():
    value = TF.Timestamp
    assert isinstance(value, datetime)
    expected = datetime(2019, 1, 1, 6, 0, tzinfo=timezone.utc).astimezone()
    expected = expected.replace(tzinfo=None)
    assert value == expected


def test_complexsingle():
    value = TF.ComplexSingle
    assert isinstance(value, complex)
    assert value == complex(1.0, 0.0)


def test_complexdouble():
    value = TF.ComplexDouble
    assert isinstance(value, complex)
    assert value == complex(0.0, -1.0)
