"""TDMS file reader class

TDMS File Format Internal Structure
http://www.ni.com/product-documentation/5696/en/
"""
from pathlib import Path
import re as regex
from functools import lru_cache
import struct
from nitdms.common import (
    LeadIn,
    Segment,
    TdsDataType,
    KToC,
    STRUCT_FORMAT,
    TdmsObject,
    Group,
    Channel,
)
from nitdms.exceptions import (
    InvalidTDMSFileError,
    InvalidTDMSVersionError,
    InvalidDimensionError,
    SegmentCorruptedError,
    DataTypeNotSupportedError,
)


class TdmsFile(TdmsObject):
    """TDMS file object containing all groups, channels, properties and data

    Upon instantiation, the class loads the TDMS file into memory and discovers
    all of the file, group and channel objects and respective properties. These
    objects and properties are dynamically instantiated as attributes allowing easy
    access from within an interactive session with tab completion such as Jupyter.

    Attributes:
        file (str): TDMS file to read

    Example:
        >>> from nitdms import TdmsFile
        >>> tf = TdmsFile(<file>)
        >>> data = tf.<group>.<channel>.data
        >>> data
        array([...])
    """

    # pylint: disable=protected-access
    def __init__(self, file):
        self._file = Path(file)
        self._name = self._file.stem
        if not self._file.is_file():
            raise FileNotFoundError(f"'{self._file}' does not exist or is not a file")
        self._buffer = None
        self._ptr = 0
        self._info = {
            "name": self._file.name,
            "file_properties": [],
            "groups": [],
            "group_properties": {},
            "group_channels": {},
            "group_channel_properties": {},
        }
        self._load()
        self._discover()

    def __iter__(self):
        groups = [gr for gr in self.__dict__ if isinstance(self[gr], Group)]
        for group in groups:
            yield self[group]

    def __repr__(self):
        attributes = [key for key in self.__dict__ if not key.startswith("_")]
        attr_str = ", ".join(attributes)
        repr_str = f"<TDMS_File {self._name}: {attr_str}>"
        if len(repr_str) > 50:
            midpt = len(repr_str) // 2
            repr_str = repr_str[0 : midpt - 1] + " ... " + repr_str[midpt + 2 :]
        return repr_str

    def _load(self):
        """Load TDMS file into memory and do basic validity check"""
        with open(self._file, "rb") as f:
            self._buffer = f.read()
        tag = struct.unpack("4s", self._buffer[0:4])[0]
        if tag != b"TDSm":
            raise InvalidTDMSFileError(f"'{self._file}' is not a TDMS file")
        toc = struct.unpack("<I", self._buffer[4:8])[0]
        byte_order = ">" if toc & KToC.BigEndian else "<"
        version = struct.unpack(byte_order + "I", self._buffer[8:12])[0]
        if version != 4713:
            raise InvalidTDMSVersionError(f"'{self._file}' is not TDMS version 2.0")

    def _unpack(self, dtype, byte_order):
        """Unpack the bytes at the current ptr position and dtype and move ptr"""

        ptr = self._ptr
        buffer = self._buffer
        try:
            size, fmt = STRUCT_FORMAT[dtype]
        except KeyError:
            raise DataTypeNotSupportedError(f"{TdsDataType(dtype)}")
        if dtype == TdsDataType.String:
            size = struct.unpack(byte_order + "I", buffer[ptr : ptr + 4])[0]
            ptr += 4
            value = struct.unpack(f"{size}s", buffer[ptr : ptr + size])[0].decode()
        elif dtype == TdsDataType.TimeStamp:
            frac, sec = struct.unpack(byte_order + fmt, buffer[ptr : ptr + size])
            if frac == 0 and sec == 0:
                # LabVIEW chose to make 0.0 mean 0.0 s relative, not absolute time
                value = 0.0
            else:
                ts = sec + frac / 2 ** 64
                value = self._convert_timestamp(ts)
        elif dtype in [TdsDataType.ComplexSingleFloat, TdsDataType.ComplexDoubleFloat]:
            re, im = struct.unpack(byte_order + fmt, buffer[ptr : ptr + size])
            value = complex(re, im)
        else:
            value = struct.unpack(byte_order + fmt, buffer[ptr : ptr + size])[0]
        ptr += size
        self._ptr = ptr
        return value

    def _discover(self):
        """Discover file, group and channel objects and properties"""
        self._ptr = 0
        for leadin in self._get_leadins():
            raw_data_size = leadin.segment_len - leadin.metadata_len
            raw_data_start = self._ptr + leadin.metadata_len
            ch_data_start = raw_data_start
            interleaved = leadin.toc & KToC.InterleavedData
            byte_order = ">" if leadin.toc & KToC.BigEndian else "<"
            n_objects = self._unpack(TdsDataType.U32, byte_order)
            for object_i in range(n_objects):
                obj_path = self._unpack(TdsDataType.String, byte_order)
                path_parts = self._get_pathparts(obj_path)
                n = len(path_parts)
                if n == 1:
                    self._ptr += 4  # raw data index
                    for name, value in self._get_properties(byte_order):
                        self._create(name, value)
                        if name not in self._info["file_properties"]:
                            self._info["file_properties"].append(name)
                elif n == 2:
                    group = path_parts[1]
                    if not hasattr(self, group):
                        self._create(group, Group(group))
                        self._info["groups"].append(group)
                        self._info["group_properties"][group] = []
                        self._info["group_channels"][group] = []
                        self._info["group_channel_properties"][group] = {}
                    self._ptr += 4
                    for name, value in self._get_properties(byte_order):
                        self._get_attr(group)._create(name, value)
                        if name not in self._info["group_properties"][group]:
                            self._info["group_properties"][group].append(name)
                else:
                    gr = path_parts[1]
                    ch = path_parts[2]
                    if not hasattr(self.__dict__[gr], ch):
                        self._get_attr(gr)._create(ch, Channel(ch))
                        self._get_attr(gr, ch)._buffer = self._buffer
                        self._info["group_channels"][gr].append(ch)
                        self._info["group_channel_properties"][gr][ch] = []
                    index_type = self._unpack(TdsDataType.U32, byte_order)
                    if index_type == 0xFFFFFFFF:
                        pass
                    elif index_type in [0x00001269, 0x00001369]:
                        self._ptr += 8
                        self._unpack(TdsDataType.U64, byte_order)
                        fcs_vector_size = self._unpack(TdsDataType.U32, byte_order)
                        self._ptr += fcs_vector_size * 20
                        rdw_vector_size = self._unpack(TdsDataType.U32, byte_order)
                        rdw = self._unpack(TdsDataType.U32, byte_order)
                        count = raw_data_size // rdw
                        self._ptr += (rdw_vector_size - 1) * 4
                        rd_sz = rdw // n_objects
                        rd_fmt = {1: "b", 2: "h", 4: "i"}[rd_sz]
                        dtype = TdsDataType.DAQmxRawData
                        if count > 0:
                            seg = Segment(
                                raw_data_start,
                                raw_data_size,
                                ch_data_start,
                                (rd_sz, rd_fmt),
                                dtype,
                                None,
                                count,
                                byte_order,
                                interleaved,
                            )
                            self._get_attr(gr, ch)._segments.append(seg)
                            ch_data_start += rd_sz
                    elif index_type == 0x0000126A:
                        # digital data
                        self._ptr += 8
                        count = self._unpack(TdsDataType.U64, byte_order)
                        fcs_vector_size = self._unpack(TdsDataType.U32, byte_order)
                        self._ptr += fcs_vector_size * 17
                        rdw_vector_size = self._unpack(TdsDataType.U32, byte_order)
                        rdw = self._unpack(TdsDataType.U32, byte_order)
                        self._ptr += (rdw_vector_size - 1) * 4
                        rd_sz = rdw
                        rd_fmt = {1: "B", 2: "H", 4: "I"}[rd_sz]
                        dtype = TdsDataType.DAQmxRawData
                        if count > 0:
                            seg = Segment(
                                raw_data_start,
                                raw_data_size,
                                ch_data_start,
                                (rd_sz, rd_fmt),
                                dtype,
                                object_i,
                                count * raw_data_size // rd_sz,
                                byte_order,
                                interleaved,
                            )
                            self._get_attr(gr, ch)._segments.append(seg)
                    elif index_type == 0:
                        last_seg = self._get_attr(gr, ch)._segments[-1]
                        count = (raw_data_size // last_seg.raw_size) * last_seg.count
                        dtype_size, _ = STRUCT_FORMAT[last_seg.dtype]
                        seg = Segment(
                            raw_data_start,
                            raw_data_size,
                            ch_data_start,
                            last_seg.rd_szfmt,
                            last_seg.dtype,
                            last_seg.offsets,
                            count,
                            byte_order,
                            interleaved,
                        )
                        if interleaved:
                            ch_data_start += dtype_size
                        else:
                            ch_data_start += count * dtype_size
                        self._get_attr(gr, ch)._segments.append(seg)
                    elif index_type == 0x00000014:
                        dtype = self._unpack(TdsDataType.U32, byte_order)
                        try:
                            dtype_size, _ = STRUCT_FORMAT[TdsDataType(dtype)]
                        except KeyError:
                            raise DataTypeNotSupportedError(f"{TdsDataType(dtype)}")
                        dim = self._unpack(TdsDataType.U32, byte_order)
                        if dim != 1:
                            raise InvalidDimensionError
                        count = self._unpack(TdsDataType.U64, byte_order)
                        seg = Segment(
                            raw_data_start,
                            raw_data_size,
                            ch_data_start,
                            STRUCT_FORMAT[TdsDataType(dtype)],
                            TdsDataType(dtype),
                            None,
                            count,
                            byte_order,
                            interleaved,
                        )
                        self._get_attr(gr, ch)._segments.append(seg)
                        if interleaved:
                            ch_data_start += dtype_size
                        else:
                            ch_data_start += count * dtype_size
                    elif index_type == 0x0000001C:
                        dtype = self._unpack(TdsDataType.U32, byte_order)
                        dim = self._unpack(TdsDataType.U32, byte_order)
                        count = self._unpack(TdsDataType.U64, byte_order)
                        raw_chunk_size = self._unpack(TdsDataType.U64, byte_order)
                        self._ptr += 4
                        raw_size = raw_chunk_size - 4 * count
                        reps = raw_data_size // raw_chunk_size
                        fmt = f"{byte_order}{count}I"
                        st = self._ptr
                        en = st + 4 * count
                        buf = self._buffer[st:en]
                        offsets = (0,) + struct.unpack(fmt, buf)
                        raw_start = [
                            raw_data_start + i * raw_chunk_size + 4 * count
                            for i in range(reps)
                        ]
                        seg = Segment(
                            raw_start,
                            raw_size,
                            raw_start,
                            STRUCT_FORMAT[TdsDataType(dtype)],
                            TdsDataType(dtype),
                            offsets,
                            count,
                            byte_order,
                            interleaved,
                        )
                        self._get_attr(gr, ch)._segments.append(seg)
                        self._ptr -= 4
                    else:
                        raise ValueError("Unknown raw data index type")
                    for n, v in self._get_properties(byte_order):
                        if index_type in [0x00001269, 0x00001369, 0x0000126A]:
                            if n.startswith("NI_"):
                                self._get_attr(gr, ch)._scales[n] = v
                        else:
                            self._get_attr(gr, ch)._create(n, v)
                            if n not in self._info["group_channel_properties"][gr][ch]:
                                self._info["group_channel_properties"][gr][ch].append(n)

    def _get_attr(self, group, channel=None):
        """Lookup attribute for Group or Channel objects"""
        if channel is None:
            return self.__dict__[group]
        return self.__dict__[group].__dict__[channel]

    def _get_leadins(self):
        """Leadin generator function"""
        while self._ptr < len(self._buffer):
            # ToC field is always little-endian U32
            toc = struct.unpack("<I", self._buffer[self._ptr + 4 : self._ptr + 8])[0]
            byte_order = ">" if toc & KToC.BigEndian else "<"
            fmt = byte_order + "QQ"
            self._ptr += 12
            leadin = LeadIn(
                toc, *struct.unpack(fmt, self._buffer[self._ptr : self._ptr + 16])
            )
            self._ptr += 16
            if self._ptr + leadin.segment_len > len(self._buffer):
                raise SegmentCorruptedError
            metadata_start = self._ptr
            yield leadin
            if self._ptr != metadata_start + leadin.metadata_len:
                self._ptr += metadata_start + leadin.metadata_len - self._ptr
            # move the pointer to beginning of next segment
            self._ptr += leadin.segment_len - leadin.metadata_len

    def _get_properties(self, byte_order):
        """Object properties generator function"""
        n_properties = self._unpack(TdsDataType.U32, byte_order)
        for _ in range(n_properties):
            name = self._unpack(TdsDataType.String, byte_order)
            dtype = self._unpack(TdsDataType.U32, byte_order)
            value = self._unpack(TdsDataType(dtype), byte_order)
            yield (name, value)

    @staticmethod
    @lru_cache(1024)
    def _get_pathparts(obj_path):
        """Split object path of form /['group']/['channel'] and make valid"""
        parts = regex.split("/(?=')", obj_path)
        return [part.strip("'").replace("/", "_") for part in parts]

    @property
    def info(self):
        """dict: names of all the groups, channels and properties in the file"""
        return self._info

    def __str__(self):
        """Tree view of the groups, channels and properties"""
        lines = [self._info["name"]]
        for fp in self._info["file_properties"]:
            lines.append(f"    {fp}")
        for group in self._info["groups"]:
            lines.append(f"    {group}")
            for gp in self._info["group_properties"][group]:
                lines.append(f"        {gp}")
            for ch in self._info["group_channels"][group]:
                lines.append(f"        {ch}")
                for cp in self._info["group_channel_properties"][group][ch]:
                    lines.append(f"            {cp}")
                lines.append("            data")
        return "\n".join(lines)

    @property
    def groups(self):
        """Groups in file"""
        return [grp._name for grp in self]

    @property
    def name(self):
        """File name"""
        return self._name
