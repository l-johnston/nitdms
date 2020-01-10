"""LabVIEW's waveform data type in python"""
from numbers import Number
import numpy as np
from unit_system import Quantity


class WaveformDT(np.ndarray):
    """Python implementation of LabVIEW's waveform data type

    LabVIEW's waveform data type has three required attributes: t0, dt, and Y.
    Additional attributes can be set and are included in the returned WaveformDT.
    WaveformDT has the function to_xy() that will generate the x-axis array from the
    t0, dt and number of samples in the Y array.

    Attributes:
        Y (array-like): data
        dt (float): wf_increment
        t0 (float or datetime): wf_start_time

    Example:
        >>> waveform = WaveformDT([1,2,3], 1, 0)
        >>> x, y = data.to_xy()
        >>> x
        array([0., 1., 2.])
        >>> y
        array([1, 2, 3])

    WaveformDT supports Matplotlib and its labeled data interface:

        >>> import matplotlib.pyplot as plt
        >>> waveform = WaveformDT([1,2,3], 1, 0)
        >>> plt.plot('x', 'y', 'r-', data=waveform)
        [<matplotlib.lines.Line2D object ... >]
        >>> plt.show()

    It is possible to set units from the unit_system package:

        >>> waveform.xunit = "s"
        >>> waveform.yunit = "m"
        >>> plt.plot(*waveform.to_xy())

    It is also possible to build a waveform from unit_system quantities:

        >>> from unit_system.predefined_units import m, s
        >>> waveform = WaveformDT([1,2,3]*m, 1*s, 0)
        >>> plt.plot(*waveform.to_xy())

    Note:
        The x-axis array will be relative time by default. For absolute time, set the
        relative parameter to False.
    """

    def __new__(cls, Y, dt, t0):
        obj = np.asarray(Y).view(cls)
        obj._t0 = t0
        obj._dt = dt
        if isinstance(Y, Quantity):
            obj._yunit = Y.unit
        if isinstance(dt, Quantity):
            obj._xunit = dt.unit
        return obj

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=invalid-name
    def __array_finalize__(self, obj):
        self._t0 = getattr(obj, "t0", None)
        self._dt = getattr(obj, "dt", None)
        self._yunit = getattr(obj, "_yunit", None)
        self._xunit = getattr(obj, "_xunit", None)

    def __repr__(self):
        t0 = self.t0 if isinstance(self.t0, Number) else 0.0
        t0 += getattr(self, "wf_start_offset", 0.0)
        dt = self.dt
        rows = []
        if self.size < 50:
            for i, sample in enumerate(self):
                rows.append(f"{t0 + i*dt:11.4e}\t{sample:11.4e}")
        else:
            for i, sample in enumerate(self[:5]):
                rows.append(f"{t0 + i*dt:11.4e}\t{sample:11.4e}")
            rows.append(" ...")
            t0 = t0 + (self.size - 5) * dt
            for i, sample in enumerate(self[-5:]):
                rows.append(f"{t0 + i*dt:11.4e}\t{sample:11.4e}")
        rows.append(f"Length: {self.size}")
        rows.append(f"t0: {self.t0}")
        rows.append(f"dt: {self.dt:11.4e}")
        return "\n".join(rows)

    @property
    def Y(self):
        """ndarray or Quantity: data array"""
        if self._yunit is None:
            y = self.view(np.ndarray)
        else:
            y = self.view(Quantity)
            y.unit = self._yunit
        return y

    @property
    def yunit(self):
        """unit (str): Y unit"""
        return self._yunit

    @yunit.setter
    def yunit(self, unit):
        self._yunit = unit

    @property
    def xunit(self):
        """unit (str): x-axis unit"""
        return self._xunit

    @xunit.setter
    def xunit(self, unit):
        self._xunit = unit

    @property
    def dt(self):
        """float or Quantity: waveform increment"""
        if self._xunit is None or isinstance(self._dt, Quantity):
            dt = self._dt
        else:
            dt = Quantity(self._dt, self._xunit)
        return dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def t0(self):
        """datetime or float: waveform start time"""
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = value

    def set_attributes(self, **kwargs):
        """Set waveform attributes"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_xy(self, relative=True):
        """Generate the (x, y) tuple

        Args:
            relative (bool): y is relative time if True, absolute if False

        Returns:
            tuple: x, y arrays
        """
        y = self.Y
        y = y.flatten()
        dt = self.dt
        t0 = self.t0
        t0_offset = getattr(self, "wf_start_offset", 0.0)
        if isinstance(dt, Quantity):
            t0_offset = Quantity(t0_offset, dt.unit)
        samples = y.size
        if relative:
            t0 = t0 if isinstance(t0, Number) else 0.0
            t0 = t0 + t0_offset
            x = np.linspace(t0, t0 + samples * dt, samples, False)
        else:
            t0 = np.datetime64(t0.astimezone().replace(tzinfo=None))
            t0_array = np.asarray([t0] * samples)
            dt = np.timedelta64(np.uint32(dt * 1e9), "ns")
            dt_array = np.asarray(
                [np.timedelta64(0, "ns")] + [dt] * (samples - 1)
            ).cumsum()
            x = t0_array + dt_array
        return (x, y)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        for input_ in inputs:
            if isinstance(input_, WaveformDT):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)
        outputs = kwargs.pop("out", None)
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, WaveformDT):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
        # pylint: disable=no-member
        # pylint complains that __array_ufunc__ is not defined in np.ndarray, but it is
        results = super(WaveformDT, self).__array_ufunc__(
            ufunc, method, *args, **kwargs
        )
        if ufunc.nout == 1:
            results = (results,)
        results = tuple(
            (np.asarray(result) if output is None else output)
            for result, output in zip(results, outputs)
        )
        # pylint: enable=no-member
        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            return None
        return results[0] if len(results) == 1 else results

    def head(self, n=5):
        """Return first n samples of the waveform

        Args:
            n (int): number of samples to return

        Returns:
            WaveformDT: first n samples
        """
        return self[:n]

    def tail(self, n=5):
        """Return the last n samples of the waveform

        Args:
            n (int): number of samples to return

        Returns:
            WaveformDT: last n samples
        """
        start_offset = self.t0 if isinstance(self.t0, Number) else 0.0
        start_offset += getattr(self, "wf_start_offset", 0.0)
        start_offset += (self.size - n) * self.dt
        wf = self[-n:]
        setattr(wf, "wf_start_offset", start_offset)
        return wf

    def __dir__(self):
        inst_attr = list(filter(lambda k: not k.startswith("_"), self.__dict__.keys()))
        cls_attr = list(filter(lambda k: not k.startswith("_"), dir(self.__class__)))
        return inst_attr + cls_attr

    def __getitem__(self, key):
        if key in ["y", "Y"]:
            return self.Y
        if key in ["x", "X"]:
            return self.to_xy()[0]
        return super().__getitem__(key)
