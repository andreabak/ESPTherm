"""Common code for base device classes"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from inspect import isabstract
from typing import Iterator, Tuple, TYPE_CHECKING, ClassVar, MutableMapping, Type, Set, List, Optional

if TYPE_CHECKING:
    from .device import Device


__all__ = [
    "iterate_type_id_paths",
    "DeviceBound",
    "DeviceTypeRegistrar",
]


def iterate_type_id_paths(base_path: str, ext: str) -> Iterator[Tuple[str, str]]:
    """
    Helper function to iterate a directory structure like `base_path/type_name/id_name.ext`
    :param base_path: the base path to start from
    :param ext: the file extension to filter `id_name` files by.
                If omitted or None, disables the filtering behaviour.
    :return: an iterator of tuples of str with (type_name, id_name)
    """
    type_name: str
    for type_name in os.listdir(base_path):
        type_dir: str = os.path.join(base_path, type_name)
        if os.path.isdir(type_dir):
            filename: str
            for filename in os.listdir(type_dir):
                filepath: str = os.path.join(type_dir, filename)
                id_name: str
                file_ext: str
                id_name, file_ext = os.path.splitext(filename)
                if os.path.isfile(filepath) and file_ext == ext:
                    yield type_name, id_name


class DeviceBound(ABC):
    """
    Abstract mixin class that keeps a read only attribute
    linking to a `Device` object set at initialization
    """
    def __init__(self, device: Device, *args, **kwargs):
        """
        Initialization for `DeviceBound`
        :param device: the `Device` object to link this instance to
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        self._device: Device = device

    @property
    def device(self) -> Device:
        """The `Device` object linked to this instance"""
        return self._device


class DeviceTypeRegistrar(ABC):
    """
    Abstract base class that provides facilities for registering other classes against device types.
    Apply this class as a base to a certain parent class to make it act as a registrar,
    with each of its concrete subclasses getting registered by their specified device type
    for that specific parent (registrar) class.
    N.B. Nested registrar classes are not supported and defining classes as such will raise errors.
    """
    _device_types_classes_registry: ClassVar[MutableMapping[Type[DeviceTypeRegistrar],
                                                            MutableMapping[str, Type[DeviceTypeRegistrar]]]] = {}
    _classes_registrars: ClassVar[MutableMapping[Type[DeviceTypeRegistrar], Type[DeviceTypeRegistrar]]] = {}

    @staticmethod
    @abstractmethod
    def _get_device_type() -> str:
        """
        Abstract static method that concrete subclasses must implement
        to return the associated device type for registration.
        N.B. if two subclasses are defined with the same device type,
             for the same abstract parent registrar class, a NameError will be raised.
        :return: the device type str
        """

    @classmethod
    def _get_registrar_bases(cls) -> List[Type[DeviceTypeRegistrar]]:
        """
        Get the list of registrar base classes for the current class.
        :return: a subclass of DeviceTypeRegistrar
        """
        base_classes_set: Set[Type] = set(cls.mro())
        registrars_set: Set[Type[DeviceTypeRegistrar]] = set(cls._device_types_classes_registry.keys())
        # noinspection PyTypeChecker
        return list(base_classes_set.intersection(registrars_set))  # type: ignore

    def __init_subclass__(cls, **kwargs):
        """
        Initializes a defined `DeviceTypeRegistrar` subclass.
        If `cls` is an abstract class, check if `DeviceTypeRegistrar` is one of its direct bases,
        and then register it as a device type registrar base class.
        If `cls` is a concrete class, get its registrar base class
        and register it for its device type in the appropriate registry.
        :raise NameError: if a subclass is being initialized with the same device type
                          as an another one within the same registrar
        :raise TypeError: if there are multiple registrars in the mro tree,
                          or a concrete subclass is defined but doesn't inherit from any registrar
        """
        super().__init_subclass__(**kwargs)

        is_abstract_cls: bool = isabstract(cls)
        device_type_registrar: bool = DeviceTypeRegistrar in cls.__bases__
        if is_abstract_cls and not device_type_registrar:
            return  # Skip

        registrar_bases: List[Type[DeviceTypeRegistrar]] = cls._get_registrar_bases()
        base_registrars_names: str = ', '.join(f'"{base.__name__}"' for base in registrar_bases)
        if device_type_registrar:
            if registrar_bases:
                raise TypeError(f'Class {cls.__name__} is defined as a device type registrar, '
                                f'but some of its base class(es) {base_registrars_names} already are registrars '
                                f'and registrars nesting is not supported')
            cls._device_types_classes_registry[cls] = {}

        elif not is_abstract_cls:
            if not registrar_bases:
                raise TypeError(f'Class {cls.__name__} is a concrete DeviceTypeRegistrar subclass, '
                                f'but none of its base classes are defined as device type registrar')
            if len(registrar_bases) > 1:
                raise TypeError(f'Class {cls.__name__} has multiple base classes defined as device type registrar: '
                                f'{base_registrars_names}')
            registrar_base: Type[DeviceTypeRegistrar] = registrar_bases[0]
            registry: MutableMapping[str, Type[DeviceTypeRegistrar]] = cls._device_types_classes_registry[registrar_base]
            device_type: str = cls._get_device_type()
            existing_device_type_class: Type[DeviceTypeRegistrar] = registry.get(device_type)
            if existing_device_type_class is not None:
                raise NameError(f'More than one {Device.__name__} class with same device type "{device_type}" defined: '
                                f'{existing_device_type_class.__name__} and {cls.__name__}')
            registry[device_type] = cls
            cls._classes_registrars[cls] = registrar_base

    @classmethod
    def _infer_registrar_cls(cls) -> Type[DeviceTypeRegistrar]:
        """
        Infers the appropriate register base class from the current class.
        :raise TypeError: if no registrar base class can be inferred
        :return: the registrar base class, subclass of DeviceTypeRegistrar
        """
        if cls in cls._device_types_classes_registry:
            return cls
        elif cls in cls._classes_registrars:
            return cls._classes_registrars.get(cls)
        else:
            registrar_bases: List[Type[DeviceTypeRegistrar]] = cls._get_registrar_bases()
            if len(registrar_bases) == 1:
                return registrar_bases[0]
        raise TypeError(f'Could not infer a registrar base class for {cls.__name__}')

    @classmethod
    def get_known_types_classes(cls) -> Iterator[Tuple[str, Type[DeviceTypeRegistrar]]]:
        """
        Iterates through all the defined `LogRecord` classes for each device type.
        :raise TypeError: if no registrar base class can be inferred
        :return: an iterator of tuples of (device_type, registered_class)
        """
        registrar_base: Optional[Type[DeviceTypeRegistrar]] = cls._infer_registrar_cls()
        registry: MutableMapping[str, Type[DeviceTypeRegistrar]] = cls._device_types_classes_registry[registrar_base]
        for k, v in registry.items():
            yield k, v

    @classmethod
    def get_class_for_device_type(cls, device_type: str) -> Type[DeviceTypeRegistrar]:
        """
        Gets the `DeviceTypeRegistrar` subclass for the given device type within the appropriate registrar.
        :param device_type: the device type for which to get the `Device` class
        :raise KeyError: if no class for the given device type or no registrar base class is found
        :return: the `DeviceTypeRegistrar` subclass for the device type
        """
        registrar_base: Optional[Type[DeviceTypeRegistrar]] = cls._infer_registrar_cls()
        registry: MutableMapping[str, Type[DeviceTypeRegistrar]] = cls._device_types_classes_registry[registrar_base]
        device_type_cls: Optional[Type[DeviceTypeRegistrar]] = registry.get(device_type)
        if device_type_cls is None:
            raise KeyError(f'No {registrar_base.__name__} subclass found for device type "{device_type}"')
        return device_type_cls

    @classmethod
    def for_type(cls, device_type: str, *args, **kwargs) -> DeviceTypeRegistrar:
        """
        Creates a `DeviceTypeRegistrar` instance using the appropriate subclass for the given device type
        :param device_type: the device type for which to instantiate the correct `DeviceTypeRegistrar` subclass
        :param args: additional positional arguments to be passed to the `DeviceTypeRegistrar` subclass' `__init__(...)`
        :param kwargs: additional keyword arguments to be passed to the `DeviceTypeRegistrar` subclass' `__init__(...)`
        :return: the instantiated `DeviceTypeRegistrar` subclass' instance
        """
        device_cls = cls.get_class_for_device_type(device_type)
        # noinspection PyArgumentList
        return device_cls(*args, **kwargs)
