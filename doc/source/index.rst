.. HeavyEdge documentation master file, created by
   sphinx-quickstart on Tue Jun  3 11:39:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
HeavyEdge documentation
***********************

.. plot:: plot-header.py
    :include-source: False

Basic package to analyze coating profile data with "heavy edge".

Usage
=====

HeavyEdge is designed to be used either as a command line program or as a Python module.

Command line
------------

Command line interface provides pre-defined subroutines to handle profile data files.
It can be invoked by::

   heavyedge <command>

Refer to help message of ``heavyedge`` for list of commands and their arguments.

The command line interface defines specific file formats.
Refer to :ref:`io` section for detailed information.

Analysis parameters
^^^^^^^^^^^^^^^^^^^

Some analysis parameters can be passed by a configuration file in YAML forma.
If a command takes such parameters, it always has an optional ``--config`` argument.
The ``--config`` argument takes a path to config file where the parameters can be specified.
Explicitly passed values take precedence over configuration file.

Python module
-------------

The Python module :mod:`heavyedge` provides functions and classes for Python runtime.
Refer to :ref:`api` section for high-level interface.

Module reference
================

.. module:: heavyedge

This section provides reference for :mod:`heavyedge` Python module.

.. _api:

Runtime API
-----------

.. automodule:: heavyedge.api
    :members:

.. _io:

Data file API
-------------

.. automodule:: heavyedge.io
    :members:

Low-level API
-------------

.. automodule:: heavyedge.wasserstein
    :members:

.. automodule:: heavyedge.segreg
    :members:

Plugin API
==========

HeavyEdge provides the following entry points for plugins.

Command line extension
----------------------

- Entry point group : ``heavyedge.commands``
- Object : Module

Registers the subcommand to ``heavyedge`` command.
To register your own command, write a module which invokes :func:`heavyedge.cli.register_command`
and register it to this entry point.

All commands registered in the same entry point are grouped together when displayed by
help message. Define ``PLUGIN_ORDER`` attribute in the module to control the displaying
order.

.. autofunction:: heavyedge.cli.register_command

.. autoclass:: heavyedge.cli.Command
    :members:

.. autoclass:: heavyedge.cli.ConfigArgumentParser
    :members:

Custom raw data type
--------------------

- Entry point group : ``heavyedge.rawdata``
- Object : Subclass of :class:`heavyedge.io.RawProfileBase`
- Affected commands : ``heavyedge prep``
