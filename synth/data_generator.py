from __future__ import annotations
import abc
import dataclasses
import random
import time
from typing import Any

from synth import primitives as P


class UnsupportedDataType(Exception):
    pass


@dataclasses.dataclass
class DataGenContext:
    int_lb: int = -10
    int_ub: int = 20

    fixed_list_len: int | None = None
    list_minlen: int = 2
    list_maxlen: int = 30


class MainDataGenerator:

    def __init__(self) -> None:
        self._list_generators: list[IDataTypeGenerator] = []

    def add_generator(self, generator: IDataTypeGenerator):
        generator.set_parent(self)
        self._list_generators.append(generator)

    def generate(self, context: DataGenContext, dtype: P.DataType) -> Any:
        for gen in self._list_generators:
            try:
                out = gen.generate(context, dtype)
            except UnsupportedDataType:
                pass
            else:
                return out


class IDataTypeGenerator:

    def __init__(self, seed: int) -> None:
        self._parent: MainDataGenerator
        self._rng = random.Random(seed)

    def set_parent(self, main: MainDataGenerator):
        self._parent = main

    @abc.abstractmethod
    def generate(self, context: DataGenContext, dtype: P.DataType):
        raise NotImplementedError


class ScalarDataTypeGenerator(IDataTypeGenerator):

    def generate(self, context: DataGenContext, dtype: P.DataType):
        if not isinstance(dtype, P.ScalarType):
            raise UnsupportedDataType
        match dtype:
            case P.ScalarType.Int:
                return self._rng.randint(context.int_lb, context.int_ub)
            case P.ScalarType.Boolean:
                return self._rng.random() > 0.5
            case _:
                raise UnsupportedDataType


class HContainerDataTypeGenerator(IDataTypeGenerator):

    def generate(self, context: DataGenContext, dtype: P.DataType):
        if not isinstance(dtype, P.HContainer):
            raise UnsupportedDataType
        list_len = (
            context.fixed_list_len
            if context.fixed_list_len
            else self._rng.randint(context.list_minlen, context.list_maxlen)
        )
        if (
            dtype.ctype is P.ContainerType.Tup
            and isinstance(dtype.item, P.HContainer)
            and dtype.item.ctype is P.ContainerType.Tup
        ):
            context.fixed_list_len = self._rng.randint(context.list_minlen, context.list_maxlen)
        container = [self._parent.generate(context, dtype=dtype.item) for _ in range(list_len)]
        match dtype.ctype:
            case P.ContainerType.Tup:
                return tuple(container)
            case P.ContainerType.Set:
                return frozenset(container)
            case _:
                raise UnsupportedDataType


class IContainerDataTypeGenerator(IDataTypeGenerator):

    def generate(self, context: DataGenContext, dtype: P.DataType):
        if not isinstance(dtype, P.IContainer):
            raise UnsupportedDataType
        return tuple(self._parent.generate(context, child) for child in dtype.items)


class GenericDataTypeGenerator(IDataTypeGenerator):

    def generate(self, context: DataGenContext, dtype: P.DataType):
        if not isinstance(dtype, P.GenericType):
            raise UnsupportedDataType
        if not isinstance(dtype._resolved_type, P.DataType):
            raise UnsupportedDataType
        return self._parent.generate(context, dtype._resolved_type)


def make_generator(seed: int | None = None):
    _seed = seed if seed else int(time.time())

    main = MainDataGenerator()
    main.add_generator(ScalarDataTypeGenerator(_seed))
    main.add_generator(HContainerDataTypeGenerator(_seed + 1))
    main.add_generator(IContainerDataTypeGenerator(_seed + 2))
    main.add_generator(GenericDataTypeGenerator(_seed + 3))

    return main
