"""Test TDMS objects and properties"""
from pytest import raises
from nitdms import TdmsFile
from nitdms.common import TdmsObject, Group, Channel

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member

TF = TdmsFile("./tests/tdms_files/objects and properties.tdms")


def test_fileobject():
    assert issubclass(TF.__class__, TdmsObject)
    assert isinstance(TF, TdmsFile)


def test_groupobject():
    assert issubclass(TF.group_0.__class__, TdmsObject)
    assert isinstance(TF.group_0, Group)


def test_channelobject():
    assert issubclass(TF.group_0.ch_0.__class__, TdmsObject)
    assert isinstance(TF.group_0.ch_0, Channel)


def test_fileproperty():
    value = TF.file_prop_0
    assert value == 0


def test_groupproperty():
    value = TF.group_0.group_prop_0
    assert value == 1


def test_channelproperty():
    value = TF.group_0.ch_0.ch_prop_0
    assert value == 2


def test_itemaccess():
    group = TF["group_0"]
    channel = group["ch_0"]
    value = channel["ch_prop_0"]
    assert value == 2


def test_attributeassignment():
    with raises(AttributeError):
        TF.group_0.ch_0.ch_prop_0 = 0


def test_deleteattribute():
    with raises(AttributeError):
        del TF.group_0.ch_0.ch_prop_0


def test_contains():
    result = "group_0" in TF
    assert result


def test_len():
    assert len(TF) == 3


def test_iter():
    results = set()
    for group in TF:
        results.add(group)
    assert results == set(["group_0"])


def test_changebyteorder():
    tf = TdmsFile("./tests/tdms_files/change byte order.tdms")
    assert tf.file_prop_0 == 0
    assert tf.file_prop_1 == 1
