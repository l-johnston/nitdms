"""Python binding to tdms.dll using ctypes

This is provisional support for 64-bit Windows, LabVIEW and Python.
"""
# pylint: disable=invalid-name
from os import getenv
import functools
from ctypes import (
    cdll,
    c_int,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_uint,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_float,
    c_double,
    c_ulonglong,
    c_void_p,
    c_bool,
    c_longdouble,
    c_size_t,
    POINTER,
    byref,
    Structure,
    c_longlong,
    c_char_p,
)
from enum import IntEnum
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
from nitdms.common import TdmsObject, Group

DLLPATH = str(
    Path(getenv("PROGRAMFILES")) / "National Instruments/Shared/TDMS/tdms.dll"
)
TDMSDLL = cdll.LoadLibrary(DLLPATH)
uInt32_MAX = 2 ** 32 - 1
uInt64_MAX = 2 ** 64 - 1
TDS_VERSION_2_0 = 4713


class CtypesEnum(IntEnum):
    """A ctypes compatible IntEnum superclass"""

    @classmethod
    def _missing_(cls, value):
        try:
            value = value.value
        except AttributeError:
            return None
        else:
            return cls.__new__(cls, value)

    @classmethod
    def from_param(cls, obj):
        """Convert enum to int"""
        if not isinstance(obj, cls):
            raise TypeError
        return int(obj)


class tdsTime(Structure):
    """LabVIEW timestamp"""

    _fields_ = [("fraction", c_ulonglong), ("sec", c_longlong)]

    def as_datetime(self):
        """Convert to datetime in machine's timezone"""
        seconds = self.sec + self.fraction / uInt64_MAX
        dt = datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        return dt.astimezone().replace(tzinfo=None)


class tdsDataType(CtypesEnum):
    """TDS data types"""

    tdsTypeVoid = 0
    tdsTypeI8 = 1
    tdsTypeI16 = 2
    tdsTypeI32 = 3
    tdsTypeI64 = 4
    tdsTypeU8 = 5
    tdsTypeU16 = 6
    tdsTypeU32 = 7
    tdsTypeU64 = 8
    tdsTypeSingleFloat = 9
    tdsTypeDoubleFloat = 10
    tdsTypeExtendedFloat = 11
    tdsTypeSingleFloatWithUnit = 0x19
    tdsTypeDoubleFloatWithUnit = 0x1A
    tdsTypeExtendedFloatWithUnit = 0x1B
    tdsTypeString = 0x20
    tdsTypeBoolean = 0x21
    tdsTypeTimeStamp = 0x44
    tdsTypeFixedPoint = 0x4F
    tdsTypeComplexSingleFloat = 0x08000C
    tdsTypeComplexDoubleFloat = 0x10000D
    tdsTypeDAQmxRawData = uInt32_MAX


as_ctype = {
    tdsDataType.tdsTypeVoid: c_void_p,
    tdsDataType.tdsTypeI8: c_int8,
    tdsDataType.tdsTypeI16: c_int16,
    tdsDataType.tdsTypeI32: c_int32,
    tdsDataType.tdsTypeI64: c_int64,
    tdsDataType.tdsTypeU8: c_uint8,
    tdsDataType.tdsTypeU16: c_uint16,
    tdsDataType.tdsTypeU32: c_uint32,
    tdsDataType.tdsTypeU64: c_uint64,
    tdsDataType.tdsTypeSingleFloat: c_float,
    tdsDataType.tdsTypeDoubleFloat: c_double,
    tdsDataType.tdsTypeExtendedFloat: c_longdouble,
    # skip floatwithunit
    tdsDataType.tdsTypeString: c_char_p,
    tdsDataType.tdsTypeBoolean: c_bool,
    tdsDataType.tdsTypeTimeStamp: tdsTime,
    tdsDataType.tdsTypeComplexSingleFloat: c_float,
    tdsDataType.tdsTypeComplexDoubleFloat: c_double,
    # skip DAQmx raw data
}


class tdsIteratorMode(CtypesEnum):
    """FileBufRead mode"""

    eTdsIteratorNone = (0,)
    eTdsIteratorByGroup = 1


class tdsFilterMode(CtypesEnum):
    """FileBufFilter mode"""

    eTdsFilterExactMatch = 0
    eTdsFilterStartWith = 1
    eTdsFilterGroupNameMatch = 2
    eTdsFilterChannelNameMatch = 3
    eTdsFilterChannelsOfGroup = 4
    eTdsFilterAllGroups = 5
    eTdsFilterPropExactMatch = 6


kfoOpen = 0
kfoOpenOrCreate = 1
kfoCreateOrReplace = 2
kfoCreate = 3
kfoReadOnly = 4
kfoMoreOptionsBigEndian = 1 << 7
kfoMoreOptionsSystemNoBuffer = 1 << 8
knsInFlagsReadMeta = 1 << 0
knsInFlagsReadIdx = 1 << 1
knsInFlagsReadRaw = 1 << 2
knsInFlagsReadChannelsOnly = 1 << 3
knsInFlagsReadJustOneObject = 1 << 4
knsInFlagsInterleavedData = 1 << 5
knsInFlagsNotNativeEndian = 1 << 6
knsOutFlagsEOF = 1 << 0
kObjTypeTdmRoot = 0
kObjTypeTdmGroup = 1
kObjTypeTdmChannel = 2
kObjTypeTdmNone = uInt32_MAX


class tdsObjType(CtypesEnum):
    """Object type"""

    kObjTypeTdmRoot = 0
    kObjTypeTdmGroup = 1
    kObjTypeTdmChannel = 2
    kObjTypeTdmNone = uInt32_MAX


def tdsapi(*args):
    """Decorator to retrieve the c-function from the dll

    Assigns the c-function to the 'call' attribute of the decorated Python function.

    Parameters
    ----------
    args : tuple containing
        function_name : str
        argtypes : tuple of ctypes
        restype : ctype, optional
    """

    def outer(py_function):
        function_name = args[0]
        argtypes = args[1]
        try:
            restype = args[2]
        except IndexError:
            restype = None
        c_function = getattr(TDMSDLL, function_name)
        if not isinstance(argtypes, (tuple, list)):
            argtypes = (argtypes,)
        c_function.argtypes = argtypes
        if restype is not None:
            c_function.restype = restype

        @functools.wraps(py_function)
        def inner(*args):
            return py_function(*args)

        inner.call = c_function
        return inner

    return outer


@tdsapi("TdsFileOpenExU", (c_char_p, c_int, c_int, c_uint32, POINTER(c_size_t)))
def fileopen(filepath, openoptions=kfoOpen, moreoptions=kfoMoreOptionsSystemNoBuffer):
    """Open a TDMS file and return a handle to it

    Parameters
    ----------
    filepath : str
    openoptions : int
    moreoptions : int

    Returns
    -------
    fileid : int
    """
    p_filepath = c_char_p(filepath.encode("utf-8"))
    fileid = c_size_t(0)
    fileopen.call(p_filepath, openoptions, moreoptions, TDS_VERSION_2_0, byref(fileid))
    return fileid.value


@tdsapi("TdsFileClose", c_size_t)
def fileclose(fileid):
    """Close a TDMS file

    Parameters
    ----------
    fileid : int
    """
    fileid = c_size_t(fileid)
    fileclose.call(fileid)


@tdsapi("TdsFileBufIteratorSetU", (c_uint, c_void_p, c_size_t))
def setiterator(mode, fileid):
    """Set iterator for buffer

    Parameters
    ----------
    mode : tdsIteratorMode
    fileid : int
    """
    setiterator.call(mode, None, fileid)


@tdsapi("TdsFileBufRead", (c_int32, POINTER(c_int32), c_size_t))
def read(inflags, fileid):
    """Read file into memory buffer

    Parameters
    ----------
    inflags : int
    fileid : int

    Returns
    -------
    outflags : int
    """
    outflags = c_int32(0)
    read.call(inflags, byref(outflags), fileid)
    return outflags.value


@tdsapi("TdsFileBufGetObjCnt", (POINTER(c_size_t), c_size_t))
def getobjcnt(fileid):
    """Get the number of objects (groups)

    Parameters
    ----------
    fileid : int

    Returns
    -------
    objcnt : int
    """
    objcnt = c_size_t(0)
    getobjcnt.call(byref(objcnt), fileid)
    return objcnt.value


@tdsapi("TdsFileBufGetFirstObjId", (POINTER(c_size_t), c_size_t))
def getfirstobjid(fileid):
    """Get the id of the first object

    Parameters
    ----------
    fileid : int

    Returns
    -------
    objid : int
    """
    objid = c_size_t(0)
    getfirstobjid.call(byref(objid), fileid)
    return objid.value


@tdsapi("TdsFileBufGetObjPathU", (POINTER(c_char_p), c_size_t, c_size_t))
def getobjpath(fileid, objid):
    """Get the path for object

    The dll allocates memory for the object path string.

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    objpath : str
    """
    objpath = c_char_p(b"\x00")
    getobjpath.call(byref(objpath), fileid, objid)
    return objpath.value.decode()


@tdsapi("TdsFileBufGetNextObjId", (POINTER(c_size_t), c_size_t, c_size_t))
def getnextobjid(fileid, current_objid):
    """Get the next object id

    Parameters
    ----------
    fileid : int
    current_objid : int

    Returns
    -------
    objid : int
    """
    objid = c_size_t(0)
    getnextobjid.call(byref(objid), fileid, current_objid)
    return objid.value


@tdsapi("TdsObjGetPropCnt", (POINTER(c_size_t), c_size_t, c_size_t))
def getpropcnt(fileid, objid):
    """Get the number of properties for the given object

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    propcnt : int
    """
    propcnt = c_size_t(0)
    getpropcnt.call(byref(propcnt), fileid, objid)
    return propcnt.value


@tdsapi(
    "TdsPropGetInfoU",
    (
        POINTER(c_char_p),
        POINTER(c_uint32),
        c_size_t,
        c_size_t,
        c_size_t,
    ),
)
def getpropinfo(fileid, objid, propid):
    """Get property info

    Parameters
    ----------
    fileid : int
    objid : int
    propid : int
        property id from 0 to propcnt

    Returns
    -------
    propinfo : tuple (name : str, data_type : tdsDataType)
    """
    name = c_char_p(b"\x00")
    dt_int = c_uint32(0)
    getpropinfo.call(byref(name), byref(dt_int), fileid, objid, propid)
    return (name.value.decode(), tdsDataType(dt_int))


@tdsapi("TdsPropGetU", (c_void_p, c_size_t, c_size_t, c_size_t))
def getprop(prop_dt, fileid, objid, propid):
    """Get the property value

    Parameters
    ----------
    prop_dt : tdsDataType
    fileid : int
    objid : int
    propid : int

    Returns
    -------
    value
    """
    value = as_ctype[prop_dt]()
    getprop.call(byref(value), fileid, objid, propid)
    value = value.value
    if isinstance(value, bytes):
        value = value.decode()
    return value


@tdsapi("TdsObjGetType", (POINTER(c_uint32), c_size_t, c_size_t))
def getobjtype(fileid, objid):
    """Get type of object

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    objtype : tdsObjType
    """
    objtype_int = c_uint32(0)
    getobjtype.call(byref(objtype_int), fileid, objid)
    return tdsObjType(objtype_int)


@tdsapi("TdsObjRawGetCntAll", (POINTER(c_ulonglong), c_size_t, c_size_t))
def getchanneldatacnt(fileid, objid):
    """Get the number of data values

    Parameters
    ----------
    fileid : int
    objid : int
        channel object id

    Returns
    -------
    cnt : int
    """
    cnt = c_ulonglong(0)
    getchanneldatacnt.call(byref(cnt), fileid, objid)
    return cnt.value


@tdsapi("TdsObjRawGetType", (POINTER(c_uint32), c_size_t, c_size_t))
def getchanneldatatype(fileid, objid):
    """Get channel data type

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    data_type : tdsDataType
    """
    dt_int = c_uint32(0)
    getchanneldatatype.call(byref(dt_int), fileid, objid)
    return tdsDataType(dt_int)


@tdsapi("TdsObjGetIdxCnt", (POINTER(c_size_t), c_size_t, c_size_t))
def getindexcnt(fileid, objid):
    """Get index count for channel data

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    idxcnt : int
    """
    idxcnt = c_size_t(0)
    getindexcnt.call(byref(idxcnt), fileid, objid)
    return idxcnt.value


@tdsapi(
    "TdsObjRawGet64",
    (c_void_p, c_ulonglong, POINTER(c_size_t), c_size_t, c_size_t, c_bool),
)
def getchanneldata(fileid, objid):
    """Get channel data

    Parameters
    ----------
    fileid : int
    objid : int

    Returns
    -------
    data
    """
    cnt = getchanneldatacnt(fileid, objid)
    dt = getchanneldatatype(fileid, objid)
    if dt == tdsDataType.tdsTypeString:
        buffer = _getstrings(cnt, fileid, objid)
    elif dt in [
        tdsDataType.tdsTypeComplexSingleFloat,
        tdsDataType.tdsTypeComplexDoubleFloat,
    ]:
        buffer = (as_ctype[dt] * 2 * cnt)()
        getchanneldata.call(buffer, 0, byref(c_size_t(cnt)), fileid, objid, True)
        buffer = [complex(real, imag) for real, imag in buffer]
    else:
        buffer = (as_ctype[dt] * cnt)()
        getchanneldata.call(buffer, 0, byref(c_size_t(cnt)), fileid, objid, True)
        if dt == tdsDataType.tdsTypeTimeStamp:
            timestamps = []
            for lv_timestamp in buffer:
                timestamps.append(lv_timestamp.as_datetime())
            buffer = timestamps
    return np.array(buffer)


@tdsapi(
    "TdsObjRawGetString64A",
    (
        POINTER(POINTER(c_char_p)),
        POINTER(POINTER(c_uint32)),
        c_size_t,
        POINTER(c_size_t),
        c_size_t,
        c_size_t,
        c_bool,
    ),
)
def _getstrings(cnt, fileid, objid):
    """Get channel data when channel data type is string

    Parameters
    ----------
    cnt : int
    fileid : int
    objid : int

    Returns
    -------
    data
    """
    buffer = (POINTER(c_char_p) * cnt)()
    sizes = (POINTER(c_uint32) * cnt)()
    strings = []
    # have to iterate for some unknown reason
    for idx in range(cnt):
        _getstrings.call(buffer, sizes, idx, byref(c_size_t(cnt)), fileid, objid, True)
        strings.append(buffer[0].contents.value.decode())
    return strings


class Channel(TdmsObject):
    """Channel object for channel properties and data

    Not intended to be instantiated in application code.
    """

    def __init__(self, name):
        self._name = name
        self._data = np.array([])

    def __iter__(self):
        for sample in self.data:
            yield sample

    @property
    def data(self):
        """ndarray or WaveformDT: the channel data"""
        return self._data

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


class TdmsFile(TdmsObject):
    """TDMS file object containing all groups, channels, properties and data

    Upon instantiation, the class loads the TDMS file into memory and discovers
    all of the file, group and channel objects and respective properties. These
    objects and properties are dynamically instantiated as attributes allowing easy
    access from within an interactive session with tab completion such as IPython.

    Parameters
    ----------
        file : str
            TDMS file to read
        mode : str
            {'open', 'open or create', 'create or replace', 'create', 'read only'}

    Example:
        >>> from nitdms import TdmsFile
        >>> tf = TdmsFile(<file>)
        >>> data = tf.<group>.<channel>.data
        >>> data
        array([...])
    """

    def __init__(self, file, mode="open"):
        self._file = Path(file)
        self._info = {
            "name": self._file.stem,
            "file_properties": {},
            "groups": [],
            "group_properties": {},
            "group_channels": {},
            "group_channel_properties": {},
        }
        if mode in ["open", "read only"]:
            if not self._file.exists():
                raise FileExistsError(self._file)
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

    def _discover(self):
        fid = fileopen(self._file.as_posix())
        read(knsInFlagsReadMeta, fid)
        for objid in range(getobjcnt(fid)):
            objpath = getobjpath(fid, objid)
            pathparts = []
            for pathpart in objpath.split("/"):
                pathparts.append(pathpart.strip("'"))
            objtype = getobjtype(fid, objid)
            if objtype is tdsObjType.kObjTypeTdmGroup:
                group = pathparts[1]
                self._info["groups"].append(group)
                self._info["group_properties"].update([[group, {}]])
                self._info["group_channels"].update([[group, []]])
                self._info["group_channel_properties"].update([[group, {}]])
                self._create(group, Group(group))
            elif objtype is tdsObjType.kObjTypeTdmChannel:
                group = pathparts[1]
                channel = pathparts[2]
                self._info["group_channels"][group].append(channel)
                self._info["group_channel_properties"][group].update([[channel, {}]])
                getattr(self, group)._create(channel, Channel(channel))
                data = getchanneldata(fid, objid)
                getattr(getattr(self, group), channel)._data = data
            for propid in range(getpropcnt(fid, objid)):
                prop, proptype = getpropinfo(fid, objid, propid)
                value = getprop(proptype, fid, objid, propid)
                if objtype is tdsObjType.kObjTypeTdmRoot:
                    self._info["file_properties"].update([[prop, value]])
                elif objtype is tdsObjType.kObjTypeTdmGroup:
                    group = pathparts[1]
                    self._info["group_properties"][group].update([[prop, value]])
                    getattr(self, group)._create(prop, value)
                else:
                    group = pathparts[1]
                    channel = pathparts[2]
                    self._info["group_channel_properties"][group][channel].update(
                        [[prop, value]]
                    )
                    getattr(getattr(self, group), channel)._create(prop, value)
        fileclose(fid)

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
