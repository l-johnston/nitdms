Usage
=====

One of the challenges in working with TDMS files is finding the properties and
channel data in a file that you didn't create. In LabVIEW code, this can be a
frustrating experience - ``nitdms`` to the rescue! At an interactive prompt, just use
tab-completion to discover all the objects available at each level of the heirarchy and
access the desired value. You can also print the file heirarchy showing all the objects.

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

