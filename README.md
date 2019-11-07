[![Build Status](https://dev.azure.com/l-johnston/nitdms/_apis/build/status/l-johnston.nitdms?branchName=master)](https://dev.azure.com/l-johnston/nitdms/_build/latest?definitionId=2&branchName=master) ![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/l-johnston/nitdms/2)
# `nitdms` - A pythonic TDMS file reader

The main export from the `nitdms` package is the TdmsFile class. Upon instantiation,
the reader loads the file into memory and discovers all of the file, group and channel
objects and respective properties. These objects and properties are dynamically 
instantiated as attributes allowing easy access from within an interactive session
with tab completion such as Jupyter or bash.

Channel data is returned as a numpy ndarray or WaveformDT if data in the file is from
LabVIEW's waveform data type. WaveformDT is a subclass of ndarray that mimics
the waveform data type.
Timestamps are datetime objects in UTC timezone.

## Installing
```bash
$ pip install nitdms
```

## Usage
Within an interactive session with tab completion:
```python
>>>from nitdms import TdmsFile
>>>tf = TdmsFile(<file>)
>>>data = tf.<group>.<channel>.data
>>>t0 = tf.<group>.<channel>.wf_start_time
>>>dt = tf.<group>.<channel>.wf_increment
>>>group_property = tf.<group>.<property>
```

Without tab completion, print a tree view to see the file hierarchy:
```python
>>>from nitdms import TdmsFile
>>>tf = TdmsFile(<file>)
>>>print(tf)
file_name
    file_prop_0
    group_0
        group_prop_0
        channel_0
            channel_prop_0
            data
>>>data = tf.group_0.channel_0.data
```

LabVIEW doesn't impose any constraints on the names of groups, channels
or properties. But, Python's attributes must be valid indentifiers - generally
ASCII letters, numbers (except first character) and underscore. So, TdmsFile also
supports item access like a dict. For example, suppose a group name in the file
is '1group' and has channel '1channel'. Both names are invalid identifiers and
will generate a syntax error when using dot access.
The usage pattern in this case is:
```python
>>>from nitdms import TdmsFile
>>>tf = TdmsFile(<file>)
>>>print(tf)
file_name
    1group
        1channel
>>>group = tf['1group']
>>>channel = group['1channel']
>>>data = channel.data
>>>
```

Want a Pandas DataFrame? For example, suppose the tdms file contains a group 'group_0'
with two channels 'ch_0' and 'ch_1' with equal length.
```python
>>>import pandas as pd
>>>from nitdms import TdmsFile
>>>tf = TdmsFile(<file>)
>>>group = tf.group_0
>>>data = dict(zip([ch for ch in group], [group[ch].data for ch in group]))
>>>df = pd.DataFrame(data)
>>>df
   ch_0   ch_1
0     0     10
1     1     11
2     2     12
...
9     9     19
>>>
```

So, why doesn't TdmsFile just return a DataFrame? The contents of the tdms file are
arbitrary and have no general, direct mapping to a DataFrame. For example, the
tdms file channel data is interpreted by the properties, but the DataFrame columns,
which are Pandas Series objects, don't support metadata. In some situations a DataFrame
is appropriate, but in general it isn't.

If the channel data in the TDMS file originated from LabVIEW's waveform data type,
the returned data will be a WaveformDT that is a subclass of numpy ndarray. This
mimics the waveform data type in LabVIEW. In addition to all of the attributes
such as t0 and dt, WaveformDT provides a convenience function to_xy() that
facilitates plotting data in matplotlib. For example:

```python
>>> import matplotlib.pyplot as plt
>>> from nitdms import TdmsFile
>>> tf = TdmsFile(<file>)
>>> data = tf.<group>.<channel>.data
>>> data.t0
datetime.datetime(...)
>>> data.dt
<value>
>>> x, y = data.to_xy()
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y)
>>> plt.show()
```

By default, to_xy() returns x-axis array as relative time. For absolute time x-axis,
set the relative parameter to False. For example:

```python
>>> import matplotlib.pyplot as plt
>>> from nitdms import TdmsFile
>>> tf = TdmsFile(<file>)
>>> data = tf.<group>.<channel>.data
>>> data.t0
datetime.datetime(...)
>>> data.dt
<value>
>>> x, y = data.to_xy(relative=False)
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y)
>>> plt.show()
```