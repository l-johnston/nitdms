"""Test example"""
from nitdms import TdmsFile

# pylint: disable=missing-docstring
# pylint: disable=pointless-statement
# pylint: disable=no-member


def test_example():
    tf = TdmsFile("./tests/tdms_files/example.tdms")
    ai0_wf = tf.analog.NI_9775_ai0.data
    assert ai0_wf.Y.shape == (4000,)
    x0, y0 = ai0_wf.to_xy()
    assert x0.size == 4000
    assert y0.size == 4000
