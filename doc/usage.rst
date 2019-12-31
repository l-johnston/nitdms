Usage
=====

One of the challenges in working with TDMS files is finding the properties and
channel data in a file that you didn't create. In LabVIEW code, this can be a
frustrating experience - ``nitdms`` to the rescue! At an interactive prompt, just use
tab-completion to discover all the objects available at each level of the heirarchy and
access the desired value.

You can also print the file heirarchy showing all the objects:

  >>> from nitdms import TdmsFile
  >>> tf = TdmsFile(<file>)
  >>> print(tf)
  <file>
    name
    <file property>
    ...
    <group>
      <group property>
      ...
      <channel>
        <channel property>
        ...
        data

LabVIEW doesn't impose any constraints on the names of groups, channels
or properties. But, Python's attributes must be valid indentifiers - generally
ASCII letters, numbers (except first character) and underscore. So, TdmsFile also
supports item access like a dict. For example, suppose a group name in the file
is '1group' and has channel '1channel'. Both names are invalid identifiers and
will generate a syntax error when using dot access.

The usage pattern in this case is:

  >>> from nitdms import TdmsFile
  >>> tf = TdmsFile(<file>)
  >>> print(tf)
  file_name
      1group
          1channel
  >>> group = tf['1group']
  >>> channel = group['1channel']
  >>> data = channel.data

Want a Pandas DataFrame? For example, suppose the tdms file contains a group 'group_0'
with two channels 'ch_0' and 'ch_1' with equal length.

  >>> import pandas as pd
  >>> from nitdms import TdmsFile
  >>> tf = TdmsFile(<file>)
  >>> group = tf.group_0
  >>> data = dict(zip([ch for ch in group], [group[ch].data for ch in group]))
  >>> df = pd.DataFrame(data)
  >>> df
    ch_0   ch_1
  0     0     10
  1     1     11
  2     2     12
  ...
  9     9     19

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

  >>> import matplotlib.pyplot as plt
  >>> from nitdms import TdmsFile
  >>> tf = TdmsFile(<file>)
  >>> waveform = tf.<group>.<channel>.data
  >>> waveform.t0
  datetime.datetime(...)
  >>> waveform.dt
  <value>
  >>> x, y = waveform.to_xy()
  >>> plt.plot(x, y)
  >>> plt.show()

WaveformDT also supports item access and Matplotlib's labeled data interface:
  >>> import matplotlib.pyplot as plt
  >>> from nitdms import TdmsFile
  >>> tf = TdmsFile(<file>)
  >>> waveform = tf.<group>.<channel>.data
  >>> plt.plot('x', 'y', 'r-', data=waveform)
  >>> plt.show()

If the channel data orginated from a DAQmx acquisition, the WaveformDT object will
have the attribute `unit_string` indicating the measurement unit. You can make the
data unit-aware using the `unit_system <https://github.com/l-johnston/unit_system>`_
package.

>>> import matplotlib.pyplot as plt
>>> from nitdms import TdmsFile
>>> from unit_system.predefined_units import *
>>> tf = TdmsFile(<file>)
>>> waveform = tf.<group>.<channel>.data
>>> waveform.unit_string
'Volts'
>>> x, y = waveform.to_xy()
>>> x = x*s
>>> y = y*V
>>> plt.plot(x, y)
>>> plt.show()