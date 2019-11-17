"""Script to generate the example"""
import numpy as np
import matplotlib.pyplot as plt
from nitdms import TdmsFile


def main():
    """generate the example graphs"""
    plt.style.use("report")
    tf = TdmsFile("cooling tower pump.tdms")
    assetname = tf.NI_CM_AssetName.split("|")[-1]
    fig, ax = plt.subplots()
    ax.set_title(assetname)
    miv_wf = tf.Waveform.MIV.data
    x, y = miv_wf.to_xy()
    ax.plot(x, y, linewidth=0.8, label="MIV")
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$t\;/\;{\rm s}$")
    ax.set_ylabel(r"$a_{\rm v}\;/\;g_{\rm n}$")
    ax.legend()
    ax.text(-0.035, -0.93, "t0 = " + miv_wf.t0.isoformat(timespec="minutes"))
    fig.savefig("miv.png")

    fig, ax = plt.subplots()
    ax.set_title(assetname)
    freq = np.fft.rfftfreq(y.size, miv_wf.dt)
    spectrum = np.fft.rfft(y)
    mag_spectrum = np.abs(spectrum) / (np.sqrt(2) * spectrum.size)
    ax.plot(freq, mag_spectrum, linewidth=0.8, label="MIV")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 0.05)
    ax.set_xlabel(r"$f\;/\;{\rm Hz}$")
    ax.set_ylabel(r"$a_{\rm v}\;/\;(g_{\rm n}\;{\rm RMS})$")
    ax.legend()
    fig.savefig("miv_fft.png")


if __name__ == "__main__":
    main()
