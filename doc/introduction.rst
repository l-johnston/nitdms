Introduction
============

A TDMS file stores data in a binary format with heirarchical structure consisting of
the file object containing one or more group objects and each group containing one
or more channel objects. Each object can contain metadata, called properties, and
channel objects can contain data such as acquisition waveforms.

::

  file object
    file properties
    group object
      group properties
      channel object
        channel properties
        data