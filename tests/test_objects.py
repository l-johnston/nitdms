"""Test TDMS objects and properties"""
from pytest import raises
from nitdms import TdmsFile
from nitdms.common import TdmsObject, Group, Channel

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member
# pylint: disable=unnecessary-comprehension
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
        results.add(group.name)
    assert results == set(["group_0"])


def test_changebyteorder():
    tf = TdmsFile("./tests/tdms_files/change byte order.tdms")
    assert tf.file_prop_0 == 0
    assert tf.file_prop_1 == 1


def test_setattrerror():
    with raises(AttributeError):
        TF["name"] = "name"


def test_deattrerror():
    with raises(AttributeError):
        del TF["name"]


def test_tdmsobjectnotimplementederror():
    to = TdmsObject()
    with raises(NotImplementedError):
        to.__repr__()
    with raises(NotImplementedError):
        to.__str__()
    with raises(NotImplementedError):
        [i for i in to]


def test_groupstr():
    group = TF.group_0
    group_str = group.__str__()
    assert group_str == "group_0\n    group_prop_0\n    ch_0"


def test_chstr():
    ch = TF.group_0.ch_0
    ch_str = ch.__str__()
    assert ch_str == "ch_0\n    ch_prop_0\n    data"


def test_filenotfound():
    with raises(FileNotFoundError):
        TdmsFile("")


def test_info():
    assert TF.info["name"] == "objects and properties.tdms"


def test_str():
    tf_str = TF.__str__()
    expected = (
        "objects and properties.tdms\n    name\n    file_prop_0\n"
        + "    group_0\n        group_prop_0\n        ch_0\n"
        + "            ch_prop_0\n            data"
    )
    assert tf_str == expected
