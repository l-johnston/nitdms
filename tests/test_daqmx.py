"""Test DAQmx channel data"""
from datetime import datetime
import numpy as np
from pytest import approx
from nitdms import TdmsFile, WaveformDT

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member


def test_daqmx_linear_voltage():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_voltage.tdms")
    data = tf.group_0.cDAQ1Mod2_ai0.data
    assert isinstance(data, WaveformDT)
    assert data.size == 10
    assert data[:3] == approx((-0.000409137, -0.000727973, -9.0301e-5))
    t0 = data.t0
    assert t0 == datetime(2019, 10, 2, 21, 31, 10, 787163)
    dt = data.dt
    assert dt == 0.001


def test_daqmx_polynomial_voltage():
    tf = TdmsFile("./tests/tdms_files/daqmx_polynomial_voltage.tdms")
    data = tf.group_0.PXI1Slot7_ai0.data
    assert data.size == 10
    assert data[:3] == approx((3.648018064, 3.647857836, 3.647857836))
    t0 = data.t0
    assert t0 == datetime(2019, 10, 2, 21, 44, 38, 544740)
    dt = data.dt
    assert dt == 0.001


def test_daqmx_resistance():
    tf = TdmsFile("./tests/tdms_files/daqmx_resistance.tdms")
    data = tf.group_0.cDAQ1Mod1_ai0.data
    assert data.size == 10
    assert data[:3] == approx((346.556966636, 346.55142377, 346.549397346))
    unit = tf.group_0.cDAQ1Mod1_ai0.unit_string
    assert unit == "Ohms"


def test_daqmx_rtd():
    tf = TdmsFile("./tests/tdms_files/daqmx_rtd.tdms")
    data = tf.group_0.cDAQ1Mod1_ai0.data
    assert data.size == 10
    assert data[:3] == approx((704.140641532, 704.121962322, 704.091729236))


def test_daqmx_rtd_below0degc():
    tf = TdmsFile("./tests/tdms_files/daqmx_rtd_below0degC.tdms")
    data = tf.group_0.cDAQ1Mod1_ai0.data
    assert data.size == 10
    assert data[:3] == approx((-128.656508921, -128.629116858, -128.628971925))


def test_daqmx_linear_current():
    tf = TdmsFile("./tests/tdms_files/daqmx_linear_current.tdms")
    data = tf.group_0.NI9208_ai0.data
    assert data.size == 10
    assert data[0] == approx(1.311302263e-8)


def test_daqmx_digital_1ch1line():
    tf = TdmsFile("./tests/tdms_files/daqmx_digital_1ch1line.tdms")
    group = tf["group_0 - line0"]
    data = group.PXI1Slot7_port0_line0.data
    result = data == np.asarray([0] * 10)
    assert result.all()


def test_daqmx_digital_1ch2lines():
    tf = TdmsFile("./tests/tdms_files/daqmx_digital_1ch2lines.tdms")
    group = tf["group_0 - line0_1"]
    data = group.PXI1Slot7_port0_line0.data
    result = data == np.asarray([0] * 10)
    assert result.all()
    data = group.PXI1Slot7_port0_line1.data
    result = data == np.asarray([1] * 10)
    assert result.all()


def test_daqmx_digital_1ch2linesboolean():
    tf = TdmsFile("./tests/tdms_files/daqmx_digital_1ch2linesboolean.tdms")
    group = tf["group_0 - line0_1"]
    data = group.PXI1Slot7_port0_line0.data
    result = data == np.asarray([0] * 10)
    assert result.all()
    data = group.PXI1Slot7_port0_line1.data
    result = data == np.asarray([1] * 10)
    assert result.all()


def test_daqmx_counter_pulsewidth():
    tf = TdmsFile("./tests/tdms_files/daqmx_counter_pulsewidth.tdms")
    assert tf.group_0.PXI1Slot7_ctr0.data == approx(10 * [0.001])
