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

The main export of nitdms is TdmsFile which is a container whose elements are attributes
that are dynamically instantiated during discovery of the TDMS file metadata.
The class supports both attribute dot access and dict-like item access.
The container is immutable and supports iteration through the attributes.