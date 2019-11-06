"""Test returning data as WaveformDT"""
from datetime import datetime, timezone
import numpy as np
from nitdms import TdmsFile, WaveformDT

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member
# pylint: disable=invalid-name
def test_analog():
    tf = TdmsFile("./tests/tdms_files/wdt_analog.tdms")
    wf = tf.group_0.ch0.data
    assert isinstance(wf, WaveformDT)
    assert wf.t0 == 0.0
    assert wf.dt == 0.01
    assert wf.size == 100


def test_analog_multiplewrites():
    tf = TdmsFile("./tests/tdms_files/wdt_analog_multiplewrites.tdms")
    wf = tf.group_0.ch0.data
    assert wf.shape == (2, 100)


def test_daqmx_linear_voltage():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_voltage.tdms")
    wf = tf.group_0.cDAQ1Mod2_ai0.data
    assert wf.dt == 0.001
    assert wf.shape == (10,)


def test_toxy_relative():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_voltage.tdms")
    wf = tf.group_0.cDAQ1Mod2_ai0.data
    dt = wf.dt
    samples = wf.size
    x = wf.to_xy()[0]
    expected = np.arange(0.0, samples * dt, dt)
    result = x == expected
    assert result.all()


def test_toxy_absolute():
    tf = TdmsFile("./tests/tdms_files/wdt_absolutet0.tdms")
    wf = tf.group_0.ch0.data
    x = wf.to_xy(relative=False)[0]
    assert wf.t0 == datetime(2019, 1, 1, 6, 0, tzinfo=timezone.utc)
    t0 = np.datetime64(datetime(2019, 1, 1))
    samples = wf.size
    dt = np.timedelta64(np.uint32(wf.dt * 1e9), "ns")
    expected = np.arange(t0, t0 + samples * dt, dt)
    result = x == expected
    assert result.all()


def test_extraattributes():
    tf = TdmsFile("./tests/tdms_files/wdt_analog.tdms")
    wf = tf.group_0.ch0.data
    assert wf.signal == "sine"


def test_repr():
    wf = WaveformDT([1, 2, 3], 1, 0)
    assert wf.__repr__() == "WaveformDT([1, 2, 3], 1, 0)"


def test_Y():
    wf = WaveformDT([1, 2, 3], 1, 0)
    results = wf.Y == np.asarray([1.0, 2.0, 3.0])
    assert results.all()


def test_ufunc():
    wf = WaveformDT([1, 2, 3], 1, 0)
    assert wf.min() == 1.0


def test_ufunc_multiply():
    wf = WaveformDT([1, 2, 3], 1, 0)
    results = 2 * wf == np.asarray([2.0, 4.0, 6.0])
    assert results.all()


def test_ufunc_multiplyat():
    wf = WaveformDT([1, 2, 3], 1, 0)
    np.multiply.at(wf, [0, 1, 2], 2.0)
    results = wf.Y == np.asarray([2.0, 4.0, 6.0])
    assert results.all()


def test_ufunc_out():
    a = WaveformDT([1.0, 2.0, 3.0], 1, 0)
    b = np.asarray([0.0] * 3)
    c = WaveformDT([0.0] * 3, 2, 1)
    np.multiply(a, 2.0, out=b)
    results = b == np.asarray([2.0, 4.0, 6.0])
    assert results.all()
    np.multiply(a, 2.0, out=c)
    results = c == WaveformDT([2.0, 4.0, 6.0], 2, 1)
    assert results.all()
