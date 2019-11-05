"""Test returning data as WaveformDT"""
from datetime import datetime, timezone
import numpy as np
from nitdms import TdmsFile, WaveformDT

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member
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
    assert wf.t0 == datetime(2019, 1, 1).astimezone(timezone.utc)
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
