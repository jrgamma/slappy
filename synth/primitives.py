from __future__ import annotations

import abc
import copy
import dataclasses
import enum
import functools
import inspect
import re
import types
from typing import TypeAlias, get_args, get_origin, get_type_hints


class DataType:

    @abc.abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError

    def is_compatible(self, other: DataType) -> bool:
        if isinstance(other, GenericType):
            return other.is_compatible(self)
        return self == other

    def resolve_generics(self, src: str, dtype: DataType) -> None:
        pass

    def get_resolvers(self, dtype: DataType, mapping: dict[str, DataType]) -> None:
        pass

    def get_depth(self) -> int:
        return 0


class ScalarType(DataType, enum.Enum):
    Int = int
    Boolean = bool
    Str = str

    def to_str(self) -> str:
        return self.value.__name__


class SpecialType(DataType, enum.Enum):
    Slice = slice
    Range = range

    def to_str(self) -> str:
        return self.value.__name__


class ContainerType(enum.Enum):
    Tup = tuple
    Set = frozenset

    def to_str(self) -> str:
        return self.value.__name__

    def is_compatible(self, other: ContainerType) -> bool:
        match self:
            # case ContainerType.Iter:
            #     return True
            # case ContainerType.Seq:
            #     return other in [ContainerType.Tup, self]
            case _:
                return self.value == other.value


@dataclasses.dataclass(frozen=True)
class HContainer(DataType):
    ctype: ContainerType
    item: DataType

    def get_depth(self) -> int:
        return self.item.get_depth() + 1

    def to_str(self) -> str:
        if self.ctype == ContainerType.Tup:
            return f"{self.ctype.to_str()}[{self.item.to_str()}, ...]"
        return f"{self.ctype.to_str()}[{self.item.to_str()}]"

    def resolve_generics(self, src: str, dtype: DataType) -> None:
        self.item.resolve_generics(src, dtype)

    def is_compatible(self, other: DataType) -> bool:
        to_test: DataType = other
        if isinstance(other, GenericType):
            if other._resolved_type:
                to_test = other._resolved_type
            else:
                return True
        if not isinstance(to_test, HContainer):
            return False
        if not self.ctype.is_compatible(to_test.ctype):
            return False
        return self.item.is_compatible(to_test.item)

    def get_resolvers(self, dtype: DataType, mapping: dict[str, DataType]) -> None:
        if isinstance(dtype, GenericType) and isinstance(
            dtype._resolved_type, HContainer
        ):
            self.get_resolvers(dtype._resolved_type, mapping)
        elif isinstance(dtype, HContainer):
            self.item.get_resolvers(dtype.item, mapping)


@dataclasses.dataclass(frozen=True)
class IContainer(DataType):
    items: tuple[DataType, ...]

    def to_str(self) -> str:
        return f"tuple[{', '.join(item.to_str() for item in self.items)}]"

    def resolve_generics(self, src: str, dtype: DataType) -> None:
        for item in self.items:
            item.resolve_generics(src, dtype)

    def is_compatible(self, other: DataType) -> bool:
        to_test: DataType = (
            other._resolved_type
            if isinstance(other, GenericType) and other._resolved_type
            else other
        )
        if isinstance(to_test, GenericType):
            return to_test.is_compatible(self)
        if not isinstance(to_test, IContainer):
            return False
        return all(a.is_compatible(b) for a, b in zip(self.items, to_test.items))

    def get_resolvers(self, dtype: DataType, mapping: dict[str, DataType]) -> None:
        if isinstance(dtype, GenericType) and isinstance(
            dtype._resolved_type, IContainer
        ):
            for a, b in zip(self.items, dtype._resolved_type.items):
                a.get_resolvers(b, mapping)
        elif isinstance(dtype, IContainer):
            for a, b in zip(self.items, dtype.items):
                a.get_resolvers(b, mapping)

    def get_depth(self) -> int:
        return max(a.get_depth() for a in self.items) + 1


@dataclasses.dataclass(frozen=True)
class FunctionType(DataType):
    atypes: tuple[DataType, ...] | GenericType
    rtype: DataType

    def get_depth(self) -> int:
        if isinstance(self.atypes, GenericType):
            return self.atypes.get_depth() + 1
        return max(a.get_depth() for a in self.atypes) + 1

    def to_str(self) -> str:
        args = (
            "[" + ", ".join(item.to_str() for item in self.atypes) + "]"
            if isinstance(self.atypes, tuple)
            else self.atypes.to_str()
        )
        return f"Callable[{args}, {self.rtype.to_str()}]"

    def is_arg_compatible(self, argn: int, other: DataType) -> bool:
        if isinstance(self.atypes, GenericType):
            return self.atypes.is_compatible(other) or other.is_compatible(self.atypes)
        # TODO : handle *args
        if argn >= len(self.atypes):
            return False
        return self.atypes[argn].is_compatible(other) or other.is_compatible(
            self.atypes[argn]
        )

    def is_ret_compatible(self, other: DataType) -> bool:
        return self.rtype.is_compatible(other) or other.is_compatible(self.rtype)

    @staticmethod
    def _is_compatible(first: FunctionType, second: FunctionType) -> bool:
        cloned = copy.deepcopy(first)
        if not cloned.apply_rettype(second.rtype):
            return False
        if isinstance(cloned.atypes, GenericType):
            return cloned.atypes.is_compatible(second)
        if not isinstance(cloned.atypes, tuple):
            return False
        if not isinstance(second.atypes, tuple) or (
            len(cloned.atypes) != len(second.atypes)
        ):
            return False
        for i, b in enumerate(second.atypes):
            if not cloned.apply_argn(i, b):
                return False
        return True

    def is_compatible(self, other: DataType) -> bool:
        if not isinstance(other, FunctionType):
            return False
        return self._is_compatible(self, other) and self._is_compatible(other, self)

    def resolve_generics(self, src: str, dtype: DataType) -> None:
        if isinstance(self.atypes, GenericType):
            self.atypes.resolve_generics(src, dtype)
        else:
            for item in self.atypes:
                item.resolve_generics(src, dtype)
        self.rtype.resolve_generics(src, dtype)

    def apply_argn(self, argn: int, other: DataType) -> bool:
        if self.is_arg_compatible(argn, other):
            if isinstance(self.atypes, GenericType):
                self.atypes.resolve_generics(self.atypes.value, other)
                return True
            mapping: dict[str, DataType] = {}
            self.atypes[argn].get_resolvers(other, mapping)
            for src, dtype in mapping.items():
                self.resolve_generics(src, dtype)
            return True
        return False

    def apply_rettype(self, other: DataType) -> bool:
        if self.is_ret_compatible(other):
            mapping: dict[str, DataType] = {}
            self.rtype.get_resolvers(other, mapping)
            other.get_resolvers(self.rtype, mapping)
            for src, dtype in mapping.items():
                self.resolve_generics(src, dtype)
            return True
        return False

    def get_resolvers(self, dtype: DataType, mapping: dict[str, DataType]) -> None:
        if isinstance(dtype, FunctionType):
            if isinstance(self.atypes, tuple) and isinstance(dtype.atypes, tuple):
                for a, b in zip(self.atypes, dtype.atypes):
                    a.get_resolvers(b, mapping)
            self.rtype.get_resolvers(dtype.rtype, mapping)


@dataclasses.dataclass
class GenericType(DataType):
    value: str

    def __post_init__(self):
        self.value = self.value.strip("~")
        self._resolved_type: DataType | None = None

    def __hash__(self) -> int:
        return hash(self.__class__) ^ (
            hash(self._resolved_type)
            if self._resolved_type is not None
            else hash("generic")
        )

    def get_depth(self) -> int:
        if self._resolved_type is not None:
            return self._resolved_type.get_depth()
        return 0

    def to_str(self) -> str:
        return (
            self.value if self._resolved_type is None else self._resolved_type.to_str()
        )

    def is_compatible(self, other: DataType) -> bool:
        if self._resolved_type is None:
            return True
        return self._resolved_type.is_compatible(other) or other.is_compatible(
            self._resolved_type
        )

    def resolve_generics(self, src: str, dtype: DataType) -> None:
        if (self._resolved_type is None) and (src == self.value):
            if isinstance(dtype, GenericType):
                if dtype._resolved_type:
                    self._resolved_type = copy.deepcopy(dtype._resolved_type)
            else:
                self._resolved_type = copy.deepcopy(dtype)

    def get_resolvers(self, dtype: DataType, mapping: dict[str, DataType]) -> None:
        if self._resolved_type is None:
            mapping.update({self.value: dtype})


@dataclasses.dataclass(frozen=True)
class NoneType(DataType):
    value: None = None

    def to_str(self):
        return "None"


# -------------------------------------------------------------------------------------------------#


class StringUtils:

    @staticmethod
    def find_enclosing_brackets(text: str) -> tuple[tuple[int, int], ...]:
        indices: list[list[int]] = []
        for i, char in enumerate(text):
            if char == "[":
                indices.append([i])
            elif char == "]":
                for item in reversed(indices):
                    if len(item) == 1:
                        item.append(i)
                        break

        return tuple(tuple(ind) for ind in indices)  # type: ignore

    @staticmethod
    def filter_outermost_brackets(
        indices: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if len(indices) < 2:
            return indices
        outermost = [indices[0]]
        for pair in indices:
            if pair[0] > outermost[-1][1]:
                outermost.append(pair)

        return tuple(item for item in outermost)

    @staticmethod
    def split_comma_outermost(text: str) -> tuple[str, ...]:
        cands = text.split(",")
        brackets = StringUtils.filter_outermost_brackets(
            StringUtils.find_enclosing_brackets(text)
        )
        all_comma_indices = functools.reduce(
            lambda x, y: x + [x[-1] + len(y) + 1], cands, [0]
        )[1:-1]
        comma_indices = list(
            filter(
                lambda x: not any(pair[0] <= x < pair[1] for pair in brackets),
                all_comma_indices,
            )
        )

        return [
            text[i : (j - 1) if isinstance(j, int) else j].strip()
            for i, j in zip([0] + comma_indices, comma_indices + [None])
        ]  # type: ignore


# -------------------------------------------------------------------------------------------------#


class IDataTypeParser(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def parse(cls, parser: RootParser, text: str) -> DataType | None:
        raise NotImplementedError


class RootParser:

    def parse(self, text: str) -> DataType:
        text = text.replace("collections.abc.", "")
        for parser in _DATATYPE_PARSERS:
            if (dtype := parser.parse(self, text)) is not None:
                return dtype
        raise ValueError(f"Unable to parse: {text}")


class FunctionTypeParser(IDataTypeParser):

    PATTERN_CALLABLE = re.compile(r"^(collections.abc.)?Callable\[(.*)\]$")
    PATTERN_ARG_RET = re.compile(r"^\[(.*)\],(.*)$")
    PATTERN_VARARG = re.compile(r"^\.\.\.,(.*)$")

    @classmethod
    def parse(cls, parser, text):
        match_callable = cls.PATTERN_CALLABLE.match(text)
        if match_callable:
            inner = match_callable.group(2).strip()
            if "Unpack" in inner:
                raise NotImplementedError("Sorry, TypeVarTuple not supported!")
            inner_split = StringUtils.split_comma_outermost(inner)
            if len(inner_split) != 2:
                raise ValueError(
                    f"should be split to arg and return type, got = {inner}, {inner_split}"
                )
            arg_str, ret_str = inner_split
            rettype = parser.parse(ret_str)
            if arg_str.startswith("[") and arg_str.endswith("]"):
                arg_types: list[DataType] = []
                for arg in StringUtils.split_comma_outermost(arg_str[1:-1]):
                    arg_types.append(parser.parse(arg.strip()))
                return FunctionType(tuple(arg_types), rettype)
            if arg_str == "...":
                return FunctionType(GenericType(arg_str), rettype)
            return None
        return None


class ContainerTypeParser(IDataTypeParser):

    PATTERN_CONTAINER = re.compile(r"^(\w+)\[(.*)\]$")

    @classmethod
    def _parse_homogenous_container(
        cls, parser: RootParser, container: str, item: str
    ) -> DataType | None:
        for ctype in ContainerType:
            if container == ctype.value.__name__:
                return HContainer(ctype, parser.parse(item))
        return None

    @classmethod
    def parse(cls, parser, text):
        if (matched := cls.PATTERN_CONTAINER.match(text)) is None:
            return None
        container_type = matched.group(1)
        item_types = StringUtils.split_comma_outermost(matched.group(2))
        if container_type == "tuple":
            if len(item_types) < 2:
                raise ValueError("Tuple items must be at least of size 2")
            if item_types[-1] != "...":
                idtypes = tuple(parser.parse(item) for item in item_types)
                return IContainer(idtypes)

        return cls._parse_homogenous_container(parser, container_type, item_types[0])


class SimpleTypeParser(IDataTypeParser):

    @classmethod
    def parse(cls, parser, text):
        for clazz in [ScalarType, SpecialType]:
            for item in clazz:
                if text == item.value.__name__:
                    return item
        return None


class GenericTypeParser(IDataTypeParser):

    @classmethod
    def parse(cls, parser, text):
        return GenericType(text)


_DATATYPE_PARSERS: list[type[IDataTypeParser]] = [
    FunctionTypeParser,
    ContainerTypeParser,
    SimpleTypeParser,
    GenericTypeParser,
]

# -------------------------------------------------------------------------------------------------#

ConstantDataType = bool | int | tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Constant:
    value: ConstantDataType
    dtype: DataType

    def to_code(self) -> str:
        if isinstance(self.value, str):
            return self.value
        return str(self.value)


@dataclasses.dataclass(frozen=True)
class TypedEntity:
    name: str
    dtype: DataType

    def to_code(self) -> str:
        return self.name

    def to_signature(self) -> str:
        return f"{self.name}: {self.dtype.to_str()}"


@dataclasses.dataclass(frozen=True)
class Arg(TypedEntity):
    caller_idx: int
    is_vararg: bool = False

    def to_signature(self):
        prefix = "*" if self.is_vararg else ""
        return prefix + super().to_signature()


@dataclasses.dataclass(frozen=True)
class _Function:
    name: str
    args: tuple[Arg, ...]
    rettype: DataType

    def get_compatible_argn(self, arg: Arg) -> list[int]:
        """Get the compatible argument number."""
        out: list[int] = []
        for i, a in enumerate(self.args):
            if a.dtype.is_compatible(arg.dtype):
                out.append(i)

        return out

    def is_return_compatible(self, other: DataType) -> bool:
        return self.rettype.is_compatible(other)

    @property
    def signature(self) -> FunctionType:
        return FunctionType(tuple(arg.dtype for arg in self.args), self.rettype)

    def to_code(self) -> str:
        return self.name

    def to_definition(self) -> str:
        ret = self.rettype.to_str()
        args = ", ".join(arg.to_signature() for arg in self.args)
        defn = f"def {self.name}({args}) -> {ret}:"
        indent = "\n" + " " * 4
        return defn + indent + self._get_body()

    def _get_body(self) -> str:
        return "..."

    @property
    def has_varag(self) -> bool:
        if len(self.args) <= 0:
            return False
        return self.args[-1].is_vararg


@dataclasses.dataclass(frozen=True)
class Variable(TypedEntity):
    value: AssgnStmt


@dataclasses.dataclass(frozen=True)
class ImportedFunction(_Function):
    module: str

    def to_code(self):
        return f"{self.module}.{super().to_code()}"


@dataclasses.dataclass(frozen=True)
class Function(_Function):
    body: tuple[AssgnStmt, ...]
    ret: ReturnStmt

    def _get_body(self) -> str:
        indent = "\n" + " " * 4
        return (
            indent.join(stmt.to_code() for stmt in self.body)
            + indent
            + self.ret.to_code()
        )


@dataclasses.dataclass(frozen=True)
class FunctionCall:
    function: _Function
    args: tuple[FuncCallArgType, ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        signature = self.function.signature
        for argn, arg in enumerate(self.args):
            if isinstance(arg, Arg | AssgnStmt | Constant):
                signature.apply_argn(argn, arg.dtype)
            elif isinstance(arg, _Function):
                signature.apply_argn(argn, arg.signature)

    def to_code(self) -> str:
        calls = ", ".join(
            arg.lhs if isinstance(arg, AssgnStmt) else arg.to_code()
            for arg in self.args
        )
        return f"{self.function.to_code()}({calls})"


@dataclasses.dataclass(frozen=True)
class AssgnStmt:
    lhs: str
    rhs: FunctionCall

    @property
    def dtype(self) -> DataType:
        return self.rhs.function.rettype

    def to_code(self) -> str:
        return f"{self.lhs} = {self.rhs.to_code()}"


@dataclasses.dataclass(frozen=True)
class ReturnStmt:
    variable: AssgnStmt

    def to_code(self) -> str:
        return f"return {self.variable.lhs}"


FuncCallArgType: TypeAlias = Arg | AssgnStmt | Constant | _Function


# -------------------------------------------------------------------------------------------------#


def get_signature(method: types.MethodType | types.FunctionType) -> inspect.Signature:

    def get_typevar_mapping(clz: object) -> dict[object, object]:
        mapping: dict[object, object] = {}
        if (bases := getattr(clz, "__orig_bases__", None)) is None:
            return mapping
        for base in bases:
            if (origin := get_origin(base)) is None:
                continue
            if getattr(origin, method.__name__, None) is None:
                break
            args = get_args(base)
            params = getattr(origin, "__parameters__", ())
            mapping.update(dict(zip(params, args)))
        return mapping

    def resolve_typevars(mapping: dict[object, object]) -> dict[str, object]:
        annotations = get_type_hints(method)
        resolved: dict[str, object] = {}
        for k, v in annotations.items():
            sv = str(v)
            for x, y in mapping.items():
                if isinstance(y, type):
                    resolved[k] = y.__qualname__
                    break
                sx = str(x)
                if sx in sv:
                    resolved[k] = sv.replace(sx, str(y))
                    break
            else:
                resolved[k] = v
        return resolved

    signature = inspect.signature(method)
    if isinstance(method, types.FunctionType):
        return signature

    mapping = get_typevar_mapping(method.__self__)
    resolved = resolve_typevars(mapping)
    return signature.replace(
        return_annotation=resolved.get("return"),
        parameters=[
            param.replace(annotation=resolved.get(param.name))
            for param in signature.parameters.values()
        ],
    )


def signature_to_function(
    func_name: str, function: types.FunctionType | types.MethodType, alias: str
) -> ImportedFunction:
    parser = RootParser()
    args: list[Arg] = []
    signature = get_signature(function)
    for argname, param in signature.parameters.items():
        dtype = parser.parse(
            param.annotation.__name__
            if not isinstance(param.annotation, str)
            and param.annotation.__class__.__module__ == "builtins"
            else str(param.annotation)
        )
        args.append(
            Arg(
                argname,
                dtype,
                param.kind
                in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL),
            )
        )
    rettype = parser.parse(
        signature.return_annotation.__name__
        if not isinstance(signature.return_annotation, str)
        and signature.return_annotation.__class__.__module__ == "builtins"
        else str(signature.return_annotation)
    )

    name = func_name
    if isinstance(function, types.MethodType):
        mod = function.__self__.__qualname__
        name = mod + "." + name

    return ImportedFunction(name, tuple(args), rettype, alias.split(".")[0])
