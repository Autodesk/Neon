"""
Neon Skeleton Configuration Module

This module provides Python bindings for Neon's skeleton configuration options,
specifically the OCC (Overlap-Communication-Computation) configuration enum.

The OCC configuration controls how computation and communication are overlapped
in Neon's skeleton execution framework.
"""

import ctypes
from enum import Enum


class SkeletonConfig():
    """
    Container class for skeleton configuration options.
    
    This class holds various configuration options for Neon's skeleton execution
    framework, including the OCC (Overlap-Communication-Computation) settings.
    """
    
    class OCC(ctypes.Structure):
        """
        OCC (Overlap-Communication-Computation) configuration class.
        
        This class represents the OCC configuration for Neon's skeleton execution
        framework. It controls how computation and communication operations are
        overlapped to optimize performance.
        
        The class extends ctypes.Structure to interface with the underlying C++
        implementation while providing a convenient Python API.
        
        Attributes:
            skeleton_occ (ctypes.c_int): The underlying C integer representation
        """
        _fields_ = [("skeleton_occ", ctypes.c_int)]

        class Values(Enum):
            """
            Enumeration of available OCC configuration values.
            
            Values:
                standard (0): Standard execution mode with basic overlap
                extended (1): Extended overlap mode with enhanced optimization
                twoWayExtended (2): Two-way extended overlap for maximum performance
                none (3): No overlap - purely sequential execution
            """
            standard = 0
            extended = 1
            twoWayExtended = 2
            none = 3


        def __init__(self, skeleton_config: 'OCC.Values'):
            """
            Initialize an OCC configuration instance.
            
            Args:
                skeleton_config (OCC.Values): The OCC configuration value to set
                
            Raises:
                No explicit validation - relies on enum membership
            """
            if skeleton_config == OCC.Values.standard:
                self.skeleton_config = ctypes.c_int(0)
            elif skeleton_config == OCC.Values.extended:
                self.skeleton_config = ctypes.c_int(1)
            elif skeleton_config == OCC.Values.twoWayExtended:
                self.skeleton_config = ctypes.c_int(2)
            elif skeleton_config == OCC.Values.none:
                self.skeleton_config = ctypes.c_int(3)

        def __str__(self):
            """
            Return a string representation of the OCC configuration.
            
            Returns:
                str: A formatted string showing memory address, size, and current value
            """
            str_repr = "<OCC: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
            if self.skeleton_config == ctypes.c_int(0):
                str_repr += f"\n\tOCC: {'standard'}"
            elif self.skeleton_config == ctypes.c_int(1):
                str_repr += f"\n\tOCC: {'extended'}"
            elif self.skeleton_config == ctypes.c_int(2):
                str_repr += f"\n\tOCC: {'twoWayExtended'}"
            elif self.skeleton_config == ctypes.c_int(3):
                str_repr += f"\n\tOCC: {'none'}"
            return str_repr

        @property
        def value(self):
            """
            Get the current OCC configuration value.
            
            Returns:
                ctypes.c_int: The underlying C integer representation
            """
            return self.skeleton_config

        @value.setter
        def value(self, skeleton_config: Values):
            """
            Set the OCC configuration value.
            
            Args:
                skeleton_config (Values): The new OCC configuration value
            """
            self.skeleton_config = skeleton_config

        @staticmethod
        def standard():
            """
            Create an OCC instance with standard configuration.
            
            Returns:
                OCC: An OCC instance configured for standard overlap mode
            """
            return OCC(OCC.Values.standard)

        @staticmethod
        def extended():
            """
            Create an OCC instance with extended configuration.
            
            Returns:
                OCC: An OCC instance configured for extended overlap mode
            """
            return OCC(OCC.Values.extended)

        @staticmethod
        def twoWayExtended():
            """
            Create an OCC instance with two-way extended configuration.
            
            Returns:
                OCC: An OCC instance configured for two-way extended overlap mode
            """
            return OCC(OCC.Values.twoWayExtended)

        @staticmethod
        def none():
            """
            Create an OCC instance with no overlap configuration.
            
            Returns:
                OCC: An OCC instance configured for sequential execution (no overlap)
            """
            return OCC(OCC.Values.none)

        @staticmethod
        def from_int(v: int):
            """
            Create an OCC instance from an integer value.
            
            Args:
                v (int): Integer value (0=standard, 1=extended, 2=twoWayExtended, 3=none)
                
            Returns:
                OCC: An OCC instance corresponding to the integer value
                
            Raises:
                Exception: If the integer value is not in the valid range [0-3]
            """
            if v == 0:
                return OCC(OCC.Values.standard)
            if v == 1:
                    return OCC(OCC.Values.extended)
            if v == 2:
                return OCC(OCC.Values.twoWayExtended)
            if v == 3:
                return OCC(OCC.Values.none)
            # raise exception
            raise Exception('Invalid OCC value')

        @staticmethod
        def from_string(s: str):
            """Create OCC instance from string value.
            
            Args:
                s: String value, one of 'standard', 'extended', 'twoWayExtended', 'none'
                
            Returns:
                OCC instance corresponding to the string value
                
            Raises:
                Exception: If the string value is not valid
            """
            if s == 'standard':
                return OCC(OCC.Values.standard)
            elif s == 'extended':
                return OCC(OCC.Values.extended)
            elif s == 'twoWayExtended':
                return OCC(OCC.Values.twoWayExtended)
            elif s == 'none':
                return OCC(OCC.Values.none)
            else:
                raise Exception(f'Invalid OCC string value: {s}. Valid options are: standard, extended, twoWayExtended, none')


        def __eq__(self, other):
            """
            Check equality between two OCC instances.
            
            Args:
                other: The other object to compare with
                
            Returns:
                bool: True if both instances have the same configuration value
                NotImplemented: If the other object is not an OCC instance
            """
            if not isinstance(other, OCC):
                return NotImplemented
            return self.skeleton_config == other.skeleton_config

