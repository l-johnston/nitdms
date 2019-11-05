"""LabVIEW's waveform data type in python"""
from numbers import Number
import numpy as np


class WaveformDT(np.ndarray):
    """Python implementation of LabVIEW's waveform data type

    Args:
        Y (array-like): data
        dt (float): wf_increment
        t0 (float or datetime): wf_start_time

    Returns:
        (WaveformDT)

    LabVIEW's waveform data type has three required attributes: t0, dt, and Y.
    Additional attributes can be set and are included in the returned WaveformDT.
    WaveformDT provides a convenience function to_xy() that facilitates plotting
    data in matplotlib. For example:

    >>> import matplotlib as plt
    >>> from nitdms import TdmsFile
    >>> tf = TdmsFile(<file>)
    >>> data = tf.<group>.<channel>.data
    >>> fig, ax = plt.subplots()
    >>> x, y = data.to_xy()
    >>> ax.plot(x, y)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.show()

    The x-axis array will be relative time by default. For absolute time, set the
    relative parameter to False.
    """

    def __new__(cls, Y, dt, t0):
        obj = np.asarray(Y).view(cls)
        obj.t0 = t0
        obj.dt = dt
        return obj

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=invalid-name
    def __array_finalize__(self, obj):
        self.t0 = getattr(obj, "t0", 0.0)
        self.dt = getattr(obj, "dt", 1.0)

    def __repr__(self):
        repr_str = super(WaveformDT, self).__repr__()
        wf_details = f", {self.dt}, {self.t0})"
        return repr_str.replace(")", wf_details)

    @property
    def Y(self):
        """Return data array"""
        return self.view(np.ndarray)

    def set_attributes(self, **kwargs):
        """Set waveform attributes"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_xy(self, relative=True):
        """Generate the (x, y) tuple

        Args:
            relative (bool): y is relative time if True, absolute if False

        Returns:
            (tuple): x, y arrays
        """
        y = self.view(np.ndarray)
        y = y.flatten()
        dt = self.dt
        t0 = self.t0
        if dt is None:
            dt = 1.0
        samples = y.size
        if relative:
            t0 = t0 if isinstance(t0, Number) else 0.0
        else:
            t0 = np.datetime64(t0.astimezone().replace(tzinfo=None))
            dt = np.timedelta64(np.uint32(dt * 1e9), "ns")
        x = np.arange(t0, t0 + samples * dt, dt)
        return (x, y)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        for input_ in inputs:
            if isinstance(input_, WaveformDT):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)
        # pylint: disable=no-member
        # pylint complains that __array_ufunc__ is not defined in np.ndarray, but it is
        results = super(WaveformDT, self).__array_ufunc__(
            ufunc, method, *args, **kwargs
        )
        # pylint: enable=no-member
        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            return None
        if ufunc.nout == 1:
            results = (results,)
        return results[0] if len(results) == 1 else results


# pylint: disable=invalid-name
if __name__ == "__main__":
    wfm = WaveformDT([1, 2, 3], 1, 0)
    print(wfm.__repr__())
