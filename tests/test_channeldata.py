"""Test channel data"""
from datetime import datetime, timezone
from nitdms import TdmsFile

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member


def test_channeldata():
    tf = TdmsFile("./tests/tdms_files/channeldata.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert len(values) == 10
    for value, expected in zip(values, range(10)):
        assert value == expected


def test_channeldata_nodata():
    tf = TdmsFile("./tests/tdms_files/channeldata_nodata.tdms")
    data = tf.group_0.ch_0.data
    assert isinstance(data, list)
    assert len(data) == 0


def test_channeldata_bigendian():
    tf = TdmsFile("./tests/tdms_files/channeldata_bigendian.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert len(values) == 10
    for value, expected in zip(values, range(10)):
        assert value == expected


def test_channeldata_2ch():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values, list)
    assert len(ch_0_values) == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values, list)
    assert len(ch_1_values) == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_interleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_interleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values, list)
    assert len(ch_0_values) == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values, list)
    assert len(ch_1_values) == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_continued():
    tf = TdmsFile("./tests/tdms_files/channeldata_continued.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert len(values) == 30
    for value, expected in zip(values, range(30)):
        assert value == expected


def test_channeldata_strings():
    tf = TdmsFile("./tests/tdms_files/channeldata_strings.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert len(values) == 30
    for value, expected in zip(values, list(map(str, range(30)))):
        assert value == expected


def test_channeldata_2groups():
    tf = TdmsFile("./tests/tdms_files/channeldata_2groups.tdms")
    group_0_values = tf.group_0.ch_0.data
    assert isinstance(group_0_values, list)
    assert len(group_0_values) == 10
    for value, expected in zip(group_0_values, range(10)):
        assert value == expected
    group_1_values = tf.group_1.ch_0.data
    assert isinstance(group_1_values, list)
    assert len(group_1_values) == 10
    for value, expected in zip(group_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_floatbigendianinterleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_floatbigendianinterleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values, list)
    assert isinstance(ch_0_values[0], float)
    assert len(ch_0_values) == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values, list)
    assert isinstance(ch_1_values[0], float)
    assert len(ch_1_values) == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_complexbigendianinterleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_complexbigendianinterleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values, list)
    assert isinstance(ch_0_values[0], complex)
    assert len(ch_0_values) == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values, list)
    assert isinstance(ch_1_values[0], complex)
    assert len(ch_1_values) == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_2ch_complex():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_complex.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values, list)
    assert isinstance(ch_0_values[0], complex)
    assert len(ch_0_values) == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values, list)
    assert isinstance(ch_1_values[0], complex)
    assert len(ch_1_values) == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_timestamp():
    tf = TdmsFile("./tests/tdms_files/channeldata_timestamp.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert isinstance(values[0], datetime)
    assert len(values) == 10
    expecteds = [
        datetime(2019, 1, 1, 0, 0, s).astimezone(timezone.utc) for s in range(10)
    ]
    for value, expected in zip(values, expecteds):
        assert value == expected


def test_channeldata_digitalwfm():
    tf = TdmsFile("./tests/tdms_files/channeldata_digitalwfm.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert isinstance(values[0], int)
    assert len(values) == 10
    expecteds = [0, 2, 3, 4, 5, 6, 7, 8, 9, 16]
    for value, expected in zip(values, expecteds):
        assert value == expected


def test_channeldata_digitalwfmstates():
    tf = TdmsFile("./tests/tdms_files/channeldata_digitalwfmstates.tdms")
    values = tf.group_0.ch_0.data
    assert isinstance(values, list)
    assert isinstance(values[0], int)
    assert len(values) == 8
    expecteds = list(range(8))
    for value, expected in zip(values, expecteds):
        assert value == expected


def test_channeldata_digitalwfmpattern():
    tf = TdmsFile("./tests/tdms_files/channeldata_digitalwfmpattern.tdms")
    group = tf.group_0
    data = [group[ch].data for ch in group]
    data.reverse()
    data = list(zip(*data))
    values = []
    for t in data:
        value = 0
        for i, bit_value in enumerate(t):
            value += 2 ** i * bit_value
        values.append(value)
    assert values == list(range(8))
    assert group.wf_start_time == 0.0
