"""Test channel data"""
from datetime import datetime
import numpy as np
from nitdms import TdmsFile, WaveformDT

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member


def test_channeldata():
    tf = TdmsFile("./tests/tdms_files/channeldata.tdms")
    data = tf.group_0.ch_0.data
    assert isinstance(data, np.ndarray)
    assert data.size == 10
    for value, expected in zip(data, range(10)):
        assert value == expected


def test_channeldata_nodata():
    tf = TdmsFile("./tests/tdms_files/channeldata_nodata.tdms")
    data = tf.group_0.ch_0.data
    assert data.size == 0


def test_channeldata_bigendian():
    tf = TdmsFile("./tests/tdms_files/channeldata_bigendian.tdms")
    data = tf.group_0.ch_0.data
    assert data.size == 10
    for value, expected in zip(data, range(10)):
        assert value == expected


def test_channeldata_2ch():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert ch_0_values.size == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert ch_1_values.size == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_interleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_interleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert ch_0_values.size == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert ch_1_values.size == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_continued():
    tf = TdmsFile("./tests/tdms_files/channeldata_continued.tdms")
    data = tf.group_0.ch_0.data
    assert data.size == 30
    for value, expected in zip(data, range(30)):
        assert value == expected


def test_channeldata_strings():
    tf = TdmsFile("./tests/tdms_files/channeldata_strings.tdms")
    data = tf.group_0.ch_0.data
    assert data.size == 30
    assert data.shape == (3, 10)
    for value, expected in zip(data.flatten(), list(map(str, range(30)))):
        assert value == expected


def test_channeldata_2groups():
    tf = TdmsFile("./tests/tdms_files/channeldata_2groups.tdms")
    group_0_values = tf.group_0.ch_0.data
    assert group_0_values.size == 10
    for value, expected in zip(group_0_values, range(10)):
        assert value == expected
    group_1_values = tf.group_1.ch_0.data
    assert group_1_values.size == 10
    for value, expected in zip(group_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_floatbigendianinterleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_floatbigendianinterleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values[0], float)
    assert ch_0_values.size == 10
    for value, expected in zip(ch_0_values, range(10)):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values[0], float)
    assert ch_1_values.size == 10
    for value, expected in zip(ch_1_values, range(10, 20)):
        assert value == expected


def test_channeldata_2ch_complexbigendianinterleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_complexbigendianinterleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values[0], complex)
    assert ch_0_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values[0], complex)
    assert ch_1_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_2ch_complex():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_complex.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values[0], complex)
    assert ch_0_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values[0], complex)
    assert ch_1_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_timestamp():
    tf = TdmsFile("./tests/tdms_files/channeldata_timestamp.tdms")
    data = tf.group_0.ch_0.data
    assert isinstance(data[0], datetime)
    assert data.size == 10
    expected = [datetime(2019, 1, 1, 0, 0, s) for s in range(10)]
    results = data == expected
    assert results.all()


def test_channeldata_digitalwfm():
    tf = TdmsFile("./tests/tdms_files/channeldata_digitalwfm.tdms")
    data = tf.group_0.ch_0.data
    assert isinstance(data, WaveformDT)
    assert isinstance(data[0], (np.int32, np.int64))
    assert data.size == 10
    expected = np.asarray([0, 2, 3, 4, 5, 6, 7, 8, 9, 16])
    results = data == expected
    assert results.all()


def test_channeldata_digitalwfmstates():
    tf = TdmsFile("./tests/tdms_files/channeldata_digitalwfmstates.tdms")
    data = tf.group_0.ch_0.data
    assert isinstance(data, WaveformDT)
    assert isinstance(data[0], (np.int32, np.int64))
    assert data.size == 8
    expected = np.asarray(list(range(8)))
    results = data == expected
    assert results.all()


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


def test_channel_iter():
    tf = TdmsFile("./tests/tdms_files/channeldata.tdms")
    for expected, value in enumerate(tf.group_0.ch_0):
        assert value == expected


def test_channeldata_timestamp_interleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_timestamp_interleaved.tdms")
    seconds = [ts.second for ts in tf.group_0.ch_0.data]
    assert seconds == list(range(0, 10, 2))
    seconds = [ts.second for ts in tf.group_0.ch_1.data]
    assert seconds == list(range(1, 10, 2))


def test_channeldata_complexsingle_interleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_complexsingle_interleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values[0], complex)
    assert ch_0_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values[0], complex)
    assert ch_1_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_2ch_complexsingle():
    tf = TdmsFile("./tests/tdms_files/channeldata_2ch_complexsingle.tdms")
    ch_0_values = tf.group_0.ch_0.data
    assert isinstance(ch_0_values[0], complex)
    assert ch_0_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10), range(19, 9, -1)))
    for value, expected in zip(ch_0_values, expecteds):
        assert value == expected
    ch_1_values = tf.group_0.ch_1.data
    assert isinstance(ch_1_values[0], complex)
    assert ch_1_values.size == 10
    expecteds = map(lambda ri: complex(*ri), zip(range(10, 20), range(9, -1, -1)))
    for value, expected in zip(ch_1_values, expecteds):
        assert value == expected


def test_channeldata_continued_interleaved():
    tf = TdmsFile("./tests/tdms_files/channeldata_continued_interleaved.tdms")
    ch_0_values = tf.group_0.ch_0.data
    expected = np.asarray(range(0, 30))
    results = ch_0_values == expected
    assert results.all()
    ch_1_values = tf.group_0.ch_1.data
    expected = []
    for i in range(3):
        expected.extend([i * 10 + j for j in range(9, -1, -1)])
    results = ch_1_values == np.asarray(expected)
    assert results.all()
