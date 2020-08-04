"""TDMS definitions

TDMS File Format Internal Structure
http://www.ni.com/product-documentation/5696/en/
"""
import math
import struct
from datetime import datetime, timedelta, timezone
from enum import Enum, IntFlag
from collections import namedtuple
import numpy as np
from nitdms.exceptions import DAQmxScaleTypeError
from nitdms.daqmx_scalers import rtdscale
from waveformDT.waveform import WaveformDT

LeadIn = namedtuple("LeadIn", ["toc", "segment_len", "metadata_len"])
Segment = namedtuple(
    "Segment",
    [
        "raw_start",
        "raw_size",
        "ch_start",
        "rd_szfmt",
        "dtype",
        "offsets",
        "count",
        "byte_order",
        "interleaved",
    ],
)


class KToC(IntFlag):
    """Lead-in table of contents (ToC) flags"""

    MetaData = 1 << 1
    NewObjectList = 1 << 2
    RawData = 1 << 3
    InterleavedData = 1 << 5
    BigEndian = 1 << 6
    DAQmxRawData = 1 << 7


class TdsDataType(Enum):
    """Data type enumeration"""

    Void = 0
    I8 = 1
    I16 = 2
    I32 = 3
    I64 = 4
    U8 = 5
    U16 = 6
    U32 = 7
    U64 = 8
    SingleFloat = 9
    DoubleFloat = 10
    ExtendedFloat = 11
    SingleFloatWithUnit = 0x19
    DoubleFloatWithUnit = 0x1A
    ExtendedFloatWithUnit = 0x1B
    String = 0x20
    Boolean = 0x21
    TimeStamp = 0x44
    FixedPoint = 0x4F
    ComplexSingleFloat = 0x08000C
    ComplexDoubleFloat = 0x10000D
    DAQmxRawData = 0xFFFFFFFF


STRUCT_FORMAT = {
    TdsDataType.I8: (1, "b"),
    TdsDataType.I16: (2, "h"),
    TdsDataType.I32: (4, "i"),
    TdsDataType.I64: (8, "q"),
    TdsDataType.U8: (1, "B"),
    TdsDataType.U16: (2, "H"),
    TdsDataType.U32: (4, "I"),
    TdsDataType.U64: (8, "Q"),
    TdsDataType.SingleFloat: (4, "f"),
    TdsDataType.DoubleFloat: (8, "d"),
    TdsDataType.String: (-1, "s"),
    TdsDataType.Boolean: (1, "?"),
    TdsDataType.TimeStamp: (16, "Qq"),
    TdsDataType.ComplexSingleFloat: (8, "ff"),
    TdsDataType.ComplexDoubleFloat: (16, "dd"),
    TdsDataType.DAQmxRawData: (4, "i"),
}


class TdmsObject:
    """Base class for the three TDMS object types: File, Group and Channel

    TdmsObject is a container whose elements are attributes that are dynamically
    instantiated during discovery of the tdms file metadata. The class, and derivatives,
    support both attribute dot access and dict-like item access.
    The container is immutable and supports iteration through the attributes.

    TdmsObject is not intended to be instantiated in application code.
    """

    def _create(self, name, value):
        """Dynamically create or update an object attribute"""
        self.__dict__[name] = value

    def __getattr__(self, name):
        try:
            value = self.__dict__[name]
        except KeyError:
            raise AttributeError(f"{name} not in {self.__repr__()}")
        else:
            return value

    def __setattr__(self, name, value):
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            raise AttributeError("can't set attribute")

    def __delattr__(self, name):
        raise AttributeError("can't delete attribute")

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __delitem__(self, name):
        self.__delattr__(name)

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __contains__(self, name):
        return hasattr(self, name)

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        attributes = [key for key in self.__dict__ if not key.startswith("_")]
        return len(attributes)

    def __dir__(self):
        attrs = super().__dir__()
        return list(filter(lambda s: not s.startswith("_"), attrs))

    @staticmethod
    def _convert_timestamp(timestamp):
        """Convert LabVIEW's timestamp to datetime

        Args:
            timestamp (float): seconds in LabVIEWS's epoch

        Returns:
            datetime: in machine's local time zone and time zone naive"""
        # LabVIEW's timestamp is UTC
        dt = datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=timestamp)
        return dt.astimezone().replace(tzinfo=None)


class Group(TdmsObject):
    """Group object for group properties and channel objects

    Not intended to be instantiated in application code.
    """

    def __init__(self, name):
        self._name = name

    def __iter__(self):
        for ch in self.channels:
            yield self[ch]

    def __repr__(self):
        attributes = [key for key in self.__dict__ if not key.startswith("_")]
        attr_str = ", ".join(attributes)
        repr_str = f"<TDMS_Group {self._name}: {attr_str}>"
        if len(repr_str) > 50:
            midpt = len(repr_str) // 2
            repr_str = repr_str[0 : midpt - 1] + " ... " + repr_str[midpt + 2 :]
        return repr_str

    def __str__(self):
        """Tree view of this group's channels and properties"""
        properties = [
            key for key in self.__dict__ if not isinstance(getattr(self, key), Channel)
        ]
        properties.remove("_name")
        channels = [
            key for key in self.__dict__ if isinstance(getattr(self, key), Channel)
        ]
        lines = [self._name]
        for gp in properties:
            lines.append(f"    {gp}")
        for ch in channels:
            lines.append(f"    {ch}")
        return "\n".join(lines)

    @property
    def channels(self):
        """Channel names in group"""
        return [ch for ch in self.__dict__ if isinstance(self[ch], Channel)]

    @property
    def name(self):
        """Group name"""
        return self._name


class Channel(TdmsObject):
    """Channel object for channel properties and data

    Not intended to be instantiated in application code.
    """

    def __init__(self, name):
        self._name = name
        self._buffer = bytes()
        self._segments = []
        self._data = None
        self._scales = {}

    def __iter__(self):
        for sample in self.data:
            yield sample

    @property
    def data(self):
        """ndarray or WaveformDT: the channel data"""
        if self._data is None:
            self._data = self._get_data()
        data = self._data
        if hasattr(self, "wf_start_time"):
            attributes = {}
            for k, v in self.__dict__.items():
                if not k.startswith("_"):
                    attributes[k] = v
            wf_start_time = attributes.pop("wf_start_time", 0.0)
            wf_increment = attributes.pop("wf_increment", 1.0)
            wf = WaveformDT(data, wf_increment, wf_start_time)
            wf.set_attributes(**attributes)
            return wf
        return data

    def _get_data(self):
        data = []
        try:
            n_values = self._segments[0].count
        except IndexError:
            n_values = 0
        uniform = True
        for segment in self._segments:
            if segment.count != n_values:
                uniform = False
            if segment.dtype == TdsDataType.String:
                for raw_start in segment.raw_start:
                    raw_size = segment.raw_size
                    buf = self._buffer[raw_start : raw_start + raw_size]
                    fmt = f"{raw_size}s"
                    string = struct.unpack(fmt, buf)[0].decode()
                    offsets = segment.offsets
                    data.extend(
                        [
                            string[offsets[i] : offsets[i + 1]]
                            for i in range(segment.count)
                        ]
                    )
            elif segment.interleaved:
                dtype_size, dtype_fmt = segment.rd_szfmt
                decimation = segment.raw_size // (dtype_size * segment.count)
                ch = (segment.ch_start - segment.raw_start) // dtype_size
                n = segment.raw_size // dtype_size
                raw_start = segment.raw_start
                raw_size = segment.raw_size
                buf = self._buffer[raw_start : raw_start + raw_size]
                if segment.dtype == TdsDataType.TimeStamp:
                    fmt = f"{segment.byte_order}{n}Q{n}Q"
                    values = struct.unpack(fmt, buf)
                    dtss = []
                    for i in range(len(values) // 2):
                        frac, sec = values[2 * i : 2 * i + 2]
                        ts = sec + frac / 2 ** 64
                        dts = self._convert_timestamp(ts)
                        dtss.append(dts)
                    data.extend(dtss[ch::decimation])
                elif segment.dtype == TdsDataType.ComplexSingleFloat:
                    fmt = f"{segment.byte_order}{n}f{n}f"
                    values = struct.unpack(fmt, buf)
                    reals = values[0::2]
                    imags = values[1::2]
                    ch_data = [complex(*(re, im)) for re, im in zip(reals, imags)]
                    data.extend(ch_data[ch::decimation])
                elif segment.dtype == TdsDataType.ComplexDoubleFloat:
                    fmt = f"{segment.byte_order}{n}d{n}d"
                    values = struct.unpack(fmt, buf)
                    reals = values[0::2]
                    imags = values[1::2]
                    ch_data = [complex(*(re, im)) for re, im in zip(reals, imags)]
                    data.extend(ch_data[ch::decimation])
                elif segment.dtype == TdsDataType.DAQmxRawData:
                    fmt = f"{segment.byte_order}{n}{dtype_fmt}"
                    values = struct.unpack(fmt, buf)[ch::decimation]
                    n_scales = self._scales["NI_Number_Of_Scales"] - 1
                    for scale in range(n_scales):
                        prefix = f"NI_Scale[{scale+1}]_"
                        scale_type = self._scales[prefix + "Scale_Type"]
                        if scale_type == "Polynomial":
                            coeff_size = self._scales[
                                prefix + "Polynomial_Coefficients_Size"
                            ]
                            scaled_values = []
                            for value in values:
                                scaled_value = 0
                                for i in range(coeff_size):
                                    coeff = self._scales[
                                        prefix + f"Polynomial_Coefficients[{i}]"
                                    ]
                                    scaled_value += coeff * math.pow(value, i)
                                scaled_values.append(scaled_value)
                        elif scale_type == "Linear":
                            slope = self._scales[prefix + "Linear_Slope"]
                            offset = self._scales[prefix + "Linear_Y_Intercept"]
                            scaled_values = [value * slope + offset for value in values]
                        elif scale_type == "RTD":
                            r0 = self._scales[prefix + "RTD_R0_Nominal_Resistance"]
                            a = self._scales[prefix + "RTD_A"]
                            b = self._scales[prefix + "RTD_B"]
                            c = self._scales[prefix + "RTD_C"]
                            i = self._scales[prefix + "RTD_Current_Excitation"]
                            lr = self._scales[prefix + "RTD_Lead_Wire_Resistance"]
                            cfg = self._scales[prefix + "RTD_Resistance_Configuration"]
                            scaled_values = []
                            for v in values:
                                t, _ = rtdscale(
                                    v, i, r0, a, b, c, lr, cfg, "DAQmx_rtdScale"
                                )
                                scaled_values.append(t)
                        else:
                            raise DAQmxScaleTypeError(f"{scale_type} not yet supported")
                        values = scaled_values
                    if fmt[-1] in ["B", "H", "I"]:
                        # digital data
                        line = segment.offsets
                        line_values = []
                        for v in values:
                            v = (v & 2 ** line) // 2 ** line
                            line_values.append(v)
                        values = line_values
                    data.extend(values)
                else:
                    fmt = f"{segment.byte_order}{n}{dtype_fmt}"
                    values = struct.unpack(fmt, buf)
                    data.extend(values[ch::decimation])
            else:
                dtype_size, dtype_fmt = segment.rd_szfmt
                ch_start = segment.ch_start
                size = dtype_size * segment.count
                buf = self._buffer[ch_start : ch_start + size]
                if segment.dtype == TdsDataType.TimeStamp:
                    n = segment.count
                    fmt = f"{segment.byte_order}{n}Q{n}Q"
                    values = struct.unpack(fmt, buf)
                    for i in range(len(values) // 2):
                        frac, sec = values[2 * i : 2 * i + 2]
                        ts = sec + frac / 2 ** 64
                        value = self._convert_timestamp(ts)
                        data.append(value)
                elif segment.dtype == TdsDataType.ComplexSingleFloat:
                    n = segment.count
                    fmt = f"{segment.byte_order}{n}f{n}f"
                    values = struct.unpack(fmt, buf)
                    reals = values[0::2]
                    imags = values[1::2]
                    data.extend([complex(*(re, im)) for re, im in zip(reals, imags)])
                elif segment.dtype == TdsDataType.ComplexDoubleFloat:
                    n = segment.count
                    fmt = f"{segment.byte_order}{n}d{n}d"
                    values = struct.unpack(fmt, buf)
                    reals = values[0::2]
                    imags = values[1::2]
                    data.extend([complex(*(re, im)) for re, im in zip(reals, imags)])
                else:
                    fmt = f"{segment.byte_order}{segment.count}{dtype_fmt}"
                    values = struct.unpack(fmt, buf)
                    data.extend(values)
        data = np.asarray(data)
        if uniform and data.size > n_values > 1:
            data = data.reshape(-1, n_values)
        return data

    def __repr__(self):
        attributes = [key for key in self.__dict__ if not key.startswith("_")]
        attr_str = ", ".join(attributes)
        repr_str = f"<TDMS_Channel {self._name}: {attr_str}>"
        if len(repr_str) > 60:
            midpt = len(repr_str) // 2
            repr_str = repr_str[0 : midpt - 1] + " ... " + repr_str[midpt + 2 :]
        return repr_str

    def __str__(self):
        """Tree view of this channel's properties and data"""
        properties = [key for key in self.__dict__ if not key.startswith("_")]
        lines = [self._name]
        for cp in properties:
            lines.append(f"    {cp}")
        lines.append("    data")
        return "\n".join(lines)

    @property
    def name(self):
        """Return name of channel"""
        return self._name
