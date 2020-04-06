"""Test returning data as WaveformDT"""
from datetime import datetime, timezone
import numpy as np
from unyt import unyt_array, s, m  # pylint: disable=no-name-in-module
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
    assert wf.Y.shape == (2, 100)


def test_daqmx_linear_voltage():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_voltage.tdms")
    wf = tf.group_0.cDAQ1Mod2_ai0.data
    assert wf.dt == 0.001
    assert wf.Y.shape == (10,)


def test_toxy_relative():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_voltage.tdms")
    wf = tf.group_0.cDAQ1Mod2_ai0.data
    dt = wf.dt
    samples = wf.size
    x = wf.to_xy()[0]
    expected = np.linspace(0.0, samples * dt, samples, False)
    result = x == expected
    assert result.all()


def test_toxy_absolute():
    tf = TdmsFile("./tests/tdms_files/wdt_absolutet0.tdms")
    wf = tf.group_0.ch0.data
    expected = datetime(2019, 1, 1, 6, 0, tzinfo=timezone.utc)
    expected = expected.astimezone().replace(tzinfo=None)
    assert wf.t0 == expected
    x, _ = wf.to_xy(False)

    def compute_expected(s):
        t = datetime(2019, 1, 1, 6, 0, s, tzinfo=timezone.utc)
        t = t.astimezone().replace(tzinfo=None)
        return np.datetime64(t)

    expected = np.asarray([compute_expected(s) for s in range(10)])
    results = x == expected
    assert results.all()


def test_extraattributes():
    tf = TdmsFile("./tests/tdms_files/wdt_analog.tdms")
    wf = tf.group_0.ch0.data
    assert wf.signal == "sine"


def test_repr():
    wf = WaveformDT([1, 2], 1, 0)
    expected = (
        " 0.0000e+00\t 1.0000e+00\n 1.0000e+00\t 2.0000e+00\n"
        + "Length: 2\nt0: 0\ndt:  1.0000e+00"
    )
    assert wf.__repr__() == expected
    wf = WaveformDT(range(60), 1, 0)
    expected = (
        " 0.0000e+00\t 0.0000e+00\n"
        + " 1.0000e+00\t 1.0000e+00\n"
        + " 2.0000e+00\t 2.0000e+00\n"
        + " 3.0000e+00\t 3.0000e+00\n"
        + " 4.0000e+00\t 4.0000e+00\n"
        + " ...\n"
        + " 5.5000e+01\t 5.5000e+01\n"
        + " 5.6000e+01\t 5.6000e+01\n"
        + " 5.7000e+01\t 5.7000e+01\n"
        + " 5.8000e+01\t 5.8000e+01\n"
        + " 5.9000e+01\t 5.9000e+01\n"
        + "Length: 60\nt0: 0\ndt:  1.0000e+00"
    )
    assert wf.__repr__() == expected


def test_Y():
    wf = WaveformDT([1, 2, 3], 1, 0)
    results = wf.Y == np.asarray([1.0, 2.0, 3.0])
    assert results.all()


def test_ufunc():
    wf = WaveformDT([1, 2, 3], 1, 0)
    x = wf.min
    assert x == 1


def test_ufunc_multiply():
    wf = WaveformDT([1, 2, 3], 1, 0)
    assert (np.asarray(2 * wf) == np.asarray([2, 4, 6])).all()


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
    assert c == WaveformDT([2.0, 4.0, 6.0], 2, 1)


def test_head():
    wf = WaveformDT(range(60), 1, 0)
    wf_head = wf.head()
    expected = (
        " 0.0000e+00\t 0.0000e+00\n"
        " 1.0000e+00\t 1.0000e+00\n"
        " 2.0000e+00\t 2.0000e+00\n"
        " 3.0000e+00\t 3.0000e+00\n"
        " 4.0000e+00\t 4.0000e+00\n"
        "Length: 5\nt0: 0\ndt:  1.0000e+00"
    )
    assert wf_head.__repr__() == expected


def test_tail():
    wf = WaveformDT(range(60), 1, 0)
    wf_tail = wf.tail()
    expected = (
        " 5.5000e+01\t 5.5000e+01\n"
        " 5.6000e+01\t 5.6000e+01\n"
        " 5.7000e+01\t 5.7000e+01\n"
        " 5.8000e+01\t 5.8000e+01\n"
        " 5.9000e+01\t 5.9000e+01\n"
        "Length: 5\nt0: 0\ndt:  1.0000e+00"
    )
    assert wf_tail.__repr__() == expected


def test_xy_item_access():
    wf = WaveformDT([1, 2, 3], 1, 0)
    x = wf["x"]
    results = x == np.asarray([0.0, 1.0, 2.0])
    assert results.all()
    y = wf["y"]
    results = y == np.asarray([1, 2, 3])
    assert results.all()


def test_wdt_from_unyt():
    wf = WaveformDT(unyt_array([1, 2, 3], "m"), unyt_array(1, "s"), 0)
    y = wf.Y
    results = y == unyt_array([1, 2, 3], "m")
    assert results.all()
    assert wf.dt == unyt_array(1, "s")


def test_wdt_to_unyt():
    wf = WaveformDT([1, 2, 3], 1, 0)
    wf.yunit = "m"
    wf.xunit = "s"
    y = wf.Y
    results = y == unyt_array([1, 2, 3], "m")
    assert results.all()
    assert wf.dt == unyt_array(1, "s")


def test_change_dt():
    wf = WaveformDT([1, 2, 3], 1, 0)
    wf.dt = 2
    assert wf.dt == 2


def test_change_t0():
    wf = WaveformDT([1, 2, 3], 1, 0)
    wf.t0 = 1
    assert wf.t0 == 1


def test_get_units():
    wf = WaveformDT(unyt_array([1, 2, 3], "m"), unyt_array(1, "s"), 0)
    assert wf.yunit == m
    assert wf.xunit == s


def test_toxy_with_units():
    wf = WaveformDT(unyt_array([1, 2, 3], "m"), unyt_array(1, "s"), 0)
    x, y = wf.to_xy()
    results = x == unyt_array([0.0, 1.0, 2.0], "s")
    assert results.all()
    results = y == unyt_array([1.0, 2.0, 3.0], "m")
    assert results.all()


def test_dir():
    wf = WaveformDT([1, 2, 3], 1, 0)
    attrs = set(dir(wf))
    assert set(["Y", "dt", "t0", "xunit", "yunit"]).issubset(attrs)
