[![Build Status](https://dev.azure.com/l-johnston/nitdms/_apis/build/status/l-johnston.nitdms?branchName=master)](https://img.shields.io/azure-devops/build/l-johnston/nitdms/14) ![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/l-johnston/nitdms/14) ![PyPI version](https://img.shields.io/pypi/v/nitdms)
# `nitdms` - A pythonic TDMS file reader

The main export from the `nitdms` package is the TdmsFile class. Upon instantiation,
the reader loads the file into memory and discovers all of the file, group and channel
objects and respective properties. These objects and properties are dynamically 
instantiated as attributes allowing easy access from within an interactive session
with tab completion such as Jupyter or bash.

Channel data is returned as a numpy ndarray, or WaveformDT if data in the file is from
LabVIEW's waveform data type. WaveformDT is a subclass of ndarray that mimics
the waveform data type.
LabVIEW's timestamp is stored as UTC in the TDMS file. `nitdms` returns the timestamp
in the machine's local time zone consistent with LabVIEW and aligns with WaveformDT.

## Installing
```bash
$ pip install nitdms
```

## Usage
Within an interactive session with tab completion:
```python
>>> from nitdms import TdmsFile
>>> tf = TdmsFile(<file>)
>>> data = tf.<group>.<channel>.data
>>> t0 = tf.<group>.<channel>.wf_start_time
>>> dt = tf.<group>.<channel>.wf_increment
>>> group_property = tf.<group>.<property>
```

## Documentation
https://l-johnston.github.io/nitdms/