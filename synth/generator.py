from __future__ import annotations

import abc
import copy
import dataclasses
import enum
import inspect
import itertools
import random
import time
import types
from collections.abc import Sequence
from typing import Generic, TypeVar

from synth import primitives as P
from synth.arc_types import ArcTypes, ARC_CONSTANTS

# --------------------------------------------------------------------------- #

_T = TypeVar("_T")
_R = TypeVar("_R")


# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FuncCompatibility:
    arg_indices: tuple[int, ...]
    ret_comp: bool = False

    def is_none(self) -> bool:
        return len(self.arg_indices) == 0 and not self.ret_comp


class Grammar:

    def __init__(
        self,
        module: types.ModuleType | None = None,
        alias: str = "",
        *,
        ignore_private: bool = True,
    ):
        self._functions: dict[str, P._Function] = {}
        if module:
            function_members = inspect.getmembers(module, inspect.isfunction)
            method_members = [
                func
                for clss in inspect.getmembers(module, inspect.isclass)
                for func in inspect.getmembers(clss[1], inspect.ismethod)
                if func[1].__module__ == module.__name__
            ]

            self._functions.update(
                {
                    member[0]: P.signature_to_function(member[0], member[1], alias)
                    for member in function_members
                }
            )
            for member in method_members:
                class_name = member[1].__self__.__qualname__
                if ignore_private and class_name.startswith("_"):
                    continue
                self._functions[f"{class_name}.{member[0]}"] = P.signature_to_function(
                    member[0], member[1], f"{alias}.{class_name}"
                )

    @property
    def functions(self) -> set[str]:
        return set(self._functions.keys())

    def get_compatible_funcs(
        self, dtype: P.DataType, *, incl_args: bool = True, incl_ret: bool = False
    ) -> dict[str, FuncCompatibility]:
        out: dict[str, FuncCompatibility] = {}
        for funcname, func in self._functions.items():
            signature = func.signature
            list_index: list[int] = []
            if incl_args:
                list_index.extend(
                    [
                        argn
                        for argn in range(len(func.args))
                        if signature.is_arg_compatible(argn, dtype)
                    ]
                )
            ret_comp = incl_ret and signature.is_ret_compatible(dtype)
            if list_index or ret_comp:
                out[funcname] = FuncCompatibility(tuple(list_index), ret_comp)

        return out

    def get_compatible_funcs_from_signature(self, dtype: P.DataType) -> list[str]:
        return [
            funcname
            for funcname, func in self._functions.items()
            if func.signature.is_compatible(dtype)
        ]

    def get(self, function: str) -> P._Function:
        return copy.deepcopy(self._functions[function])

    def add(self, function: P._Function) -> bool:
        if function.name in self._functions:
            return False
        self._functions[function.name] = function
        return True

    def get_compatible_functions_from_io_types(
        self, input_type: P.DataType, output_type: P.DataType
    ) -> dict[str, FuncCompatibility]:
        input_compatible = self.get_compatible_funcs(
            input_type, incl_args=True, incl_ret=False
        )
        output_compatible = self.get_compatible_funcs(
            output_type, incl_args=False, incl_ret=True
        )

        both_compatible: dict[str, tuple[int, ...]] = {}
        for func_in, comp_in in input_compatible.items():
            if func_in not in output_compatible:
                continue
            cand_indices: list[int] = []
            for index in comp_in.arg_indices:
                func = self.get(func_in)
                sig = func.signature
                if sig.apply_argn(index, input_type) and sig.is_ret_compatible(
                    output_type
                ):
                    cand_indices.append(index)
                elif sig.apply_rettype(output_type) and sig.is_arg_compatible(
                    index, input_type
                ):
                    cand_indices.append(index)

            if cand_indices:
                both_compatible[func_in] = tuple(cand_indices)

        return {k: FuncCompatibility(v, True) for k, v in both_compatible.items()}


# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Context:
    grammar: Grammar = dataclasses.field(default_factory=Grammar)
    local_grammar: Grammar = dataclasses.field(default_factory=Grammar)
    statements: list[P.AssgnStmt] = dataclasses.field(default_factory=list)
    input_args: list[P.Arg] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self._func_count = 0
        self._arg_count = 0

    def next_varname(self) -> str:
        return f"x{int(len(self.statements))}"

    def next_funcname(self) -> str:
        out = f"f{self._func_count}"
        self._func_count += 1
        return out

    def next_argname(self) -> str:
        out = f"a{self._arg_count}"
        self._arg_count += 1
        return out

    def get_compatible_functions(
        self, dtype: P.DataType, *, incl_args: bool = True, incl_ret: bool = False
    ) -> dict[str, FuncCompatibility]:
        out = self.grammar.get_compatible_funcs(
            dtype, incl_args=incl_args, incl_ret=incl_ret
        )
        out.update(
            self.local_grammar.get_compatible_funcs(
                dtype, incl_args=incl_args, incl_ret=incl_ret
            )
        )
        return out

    def get_compatible_functions_from_signature(self, dtype: P.DataType) -> list[str]:
        return self.grammar.get_compatible_funcs_from_signature(
            dtype
        ) + self.local_grammar.get_compatible_funcs_from_signature(dtype)

    def get_compatible_functions_from_io_types(
        self, input_type: P.DataType, return_type: P.DataType
    ) -> dict[str, FuncCompatibility]:
        out = self.grammar.get_compatible_functions_from_io_types(
            input_type, return_type
        )
        out.update(
            self.local_grammar.get_compatible_functions_from_io_types(
                input_type, return_type
            )
        )
        return out

    def get_function(self, func_name: str) -> P._Function:
        try:
            return self.grammar.get(func_name)
        except KeyError:
            return self.local_grammar.get(func_name)


class BaseSampler(abc.ABC, Generic[_T]):

    def __init__(self):
        self._rng = random.Random()

    @abc.abstractmethod
    def sample(self, context: Context) -> _T:
        raise NotImplementedError


class IntSampler(BaseSampler[int]):

    def __init__(self, lb: int, ub: int):
        super().__init__()
        self._lb = lb
        self._ub = ub

    def sample(self, context: Context) -> int:
        del context
        return self._rng.randint(self._lb, self._ub)


class BoolSampler(BaseSampler[bool]):

    def __init__(self, p: float):
        super().__init__()
        self._p = max(min(p, 1), 0)

    def sample(self, context):
        del context
        return self._rng.random() < self._p


class SequenceSampler(BaseSampler[_T]):

    def __init__(self, sequence: Sequence[_T]):
        super().__init__()
        self._seq = sequence
        self._int_sampler = IntSampler(0, len(sequence) - 1)

    def sample(self, context):
        index = self._int_sampler.sample(context)
        return self._seq[index]


class DictSampler(BaseSampler[tuple[_T, _R]]):

    def __init__(self, the_dict: dict[_T, _R]):
        super().__init__()
        self._it = the_dict
        self._int_sampler = IntSampler(0, len(the_dict) - 1)

    def sample(self, context):
        index = self._int_sampler.sample(context)
        key = list(self._it.keys())[index]
        return key, self._it[key]


# --------------------------------------------------------------------------- #


class ConstantGenerator:

    def __init__(self, stype: P.ScalarType) -> None:
        self._stype = stype
        self._sampler = BoolSampler(0.5)

    def generate(self, ctx: Context) -> P.Constant:
        # FIXME: proper implementation !!!
        out = self._sampler.sample(ctx)
        match self._stype:
            case P.ScalarType.Int:
                return P.Constant(int(out), self._stype)
            case P.ScalarType.Boolean:
                return P.Constant(out, self._stype)
            case _:
                return P.Constant("hello", self._stype)


# --------------------------------------------------------------------------- #


class DataTypeUnavailable(Exception):
    pass


class TooMuchNesting(Exception):
    pass


class ArgTypeSources(enum.IntFlag):
    input_args = enum.auto()
    func_return = enum.auto()
    variables = enum.auto()
    constants = enum.auto()
    functions = enum.auto()

    def get_choices(
        self, *, weights: dict[ArgTypeSources, int] | None = None
    ) -> list[ArgTypeSources]:
        weight = weights if weights else {}
        choices = [
            item
            for item in self.__class__
            if item & self
            for _ in range(weight.get(item, 1))
        ]
        return choices

    def sample_choice(
        self, context: Context, *, weights: dict[ArgTypeSources, int] | None = None
    ) -> ArgTypeSources:
        choices = self.get_choices(weights=weights)
        sampler = SequenceSampler(choices)
        return sampler.sample(context)


SamplingDataTypes = TypeVar(
    "SamplingDataTypes",
    str,
    P.DataType,
    ArgTypeSources,
    P.Arg,
    P.AssgnStmt,
    P.ConstantDataType,
)


class SamplingTypes(enum.IntEnum):

    io_data_types = enum.auto()
    imported_function = enum.auto()
    arg_source = enum.auto()
    constant = enum.auto()
    arg_to_apply = enum.auto()
    statements = enum.auto()


class Sampler:

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed if seed else int(time.time())
        self._rng = random.Random(self._seed)

    def reseed(self, seed: int):
        self._seed = seed

    def sample(
        self, choices: list[SamplingDataTypes], stype: SamplingTypes
    ) -> SamplingDataTypes:
        return self._rng.choice(choices)

    def reset_state(self):
        self._seed += 1
        self._rng.seed(self._seed)


class FunctionGenerator:

    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._arc_types = [t.value for t in ArcTypes]

        self._funccall_count = 0
        self._argcall_count = 0

    def _sample_dtype(self) -> P.DataType:
        return self._sampler.sample(self._arc_types, SamplingTypes.io_data_types)

    def _create_add_statement(
        self, context: Context, fc: P.FunctionCall
    ) -> P.AssgnStmt:
        stmt = P.AssgnStmt(context.next_varname(), fc)
        context.statements.append(stmt)
        return stmt

    def _sample_function_from_compat(
        self, context: Context, compats: dict[str, FuncCompatibility]
    ) -> P._Function:
        # NOTE: Decision #1
        function_names = list(compats.keys())
        func_name = self._sampler.sample(
            function_names, SamplingTypes.imported_function
        )
        func = context.get_function(func_name)
        return func

    def _sample_function_from_names(self, context: Context, namelist: list[str]):
        # NOTE: Decision #1
        func_name = self._sampler.sample(namelist, SamplingTypes.imported_function)
        func = context.get_function(func_name)
        return func

    def _generate_arg_from_dtype(
        self, context: Context, target_dtype: P.DataType, depth: int
    ) -> P.FuncCallArgType | None:
        self._argcall_count += 1

        potential_sources = ArgTypeSources(0)
        matching_constants = None
        if matching_args := [
            arg for arg in context.input_args if arg.dtype.is_compatible(target_dtype)
        ]:
            potential_sources |= ArgTypeSources.input_args
        if context.get_compatible_functions(
            target_dtype, incl_args=False, incl_ret=True
        ):
            potential_sources |= ArgTypeSources.func_return
        if matching_signatures := context.get_compatible_functions_from_signature(
            target_dtype
        ):
            potential_sources |= ArgTypeSources.functions
        if matching_variables := [
            var for var in context.statements if var.dtype.is_compatible(target_dtype)
        ]:
            potential_sources |= ArgTypeSources.variables
        if target_dtype in (
            ArcTypes.Bool.value,
            ArcTypes.Int.value,
            ArcTypes.IntPair.value,
        ):
            if (matching_constants := ARC_CONSTANTS.get(target_dtype)) is not None:
                potential_sources |= ArgTypeSources.constants

        if potential_sources == ArgTypeSources(0):
            return None

        # NOTE: Decision #1
        choices = potential_sources.get_choices(
            weights={ArgTypeSources.functions: len(matching_signatures)}
        )
        choice = self._sampler.sample(choices, SamplingTypes.arg_source)

        depth = target_dtype.get_depth()
        if depth > 5:
            raise TooMuchNesting(f"target_dtype: '{target_dtype.to_str()}'")

        match choice:
            case ArgTypeSources.input_args:
                # NOTE: Decision #2
                arg = self._sampler.sample(matching_args, SamplingTypes.arg_to_apply)
                return arg
            case ArgTypeSources.func_return:
                if fc := self._generate_funccall_from_rettype(
                    context, rettype=target_dtype, depth=depth + 1
                ):
                    stmt = self._create_add_statement(context, fc)
                    return stmt
            case ArgTypeSources.functions:
                func = self._sample_function_from_names(context, matching_signatures)
                return func
            case ArgTypeSources.variables:
                # NOTE: Decision #2
                var = self._sampler.sample(matching_variables, SamplingTypes.statements)
                return var
            case ArgTypeSources.constants:
                # NOTE: Decision #2
                if matching_constants:
                    choices = [it.value for it in matching_constants]
                    constant = P.Constant(
                        self._sampler.sample(choices, SamplingTypes.constant),
                        dtype=target_dtype,
                    )
                    return constant

    def _generate_funccall_from_rettype(
        self, context: Context, rettype: P.DataType, depth: int
    ) -> P.FunctionCall | None:
        self._funccall_count += 1

        if depth > 5 or self._funccall_count > 30:
            raise TooMuchNesting(
                f"funccall depth : {depth}, call count: {self._funccall_count}"
            )

        # get list of compatible functions
        if not (
            compats := context.get_compatible_functions(
                rettype, incl_args=False, incl_ret=True
            )
        ):
            return None

        # NOTE: #1 Sample the function
        func_to_use = self._sample_function_from_compat(context, compats)
        assert func_to_use.signature.apply_rettype(rettype)

        args_to_use: list[P.FuncCallArgType] = []
        for idx in range(len(func_to_use.args)):
            arg = func_to_use.args[idx]
            target = (
                arg.dtype._resolved_type
                if isinstance(arg.dtype, P.GenericType) and arg.dtype._resolved_type
                else arg.dtype
            )
            if not (
                arg_to_use := self._generate_arg_from_dtype(
                    context, target, depth=depth
                )
            ):
                return None

            if isinstance(arg_to_use, (P.Arg | P.AssgnStmt | P.Constant)):
                dt = arg_to_use.dtype
            elif isinstance(arg_to_use, P._Function):
                dt = arg_to_use.signature
            else:
                return None

            before = func_to_use.signature.to_str()
            state = func_to_use.signature.apply_argn(idx, dt)
            if not state:
                print(
                    f"  -> call count: {self._funccall_count}"
                    f"\n      func      : {func_to_use.name}"
                    f"\n      idx       : {idx}"
                    f"\n      sigbefore : '{before}'"
                    f"\n      signature : '{func_to_use.signature.to_str()}'"
                    f"\n      rettype   : '{func_to_use.rettype.to_str()}'"
                    f"\n      func_arg  : '{arg.dtype.to_str()}'"
                    f"\n      dt_str    : '{dt.to_str()}'"
                    f"\n      dt_repr   : {dt}"
                    f"\n      source    : {arg_to_use.__class__.__name__}"
                )
                raise AssertionError("func_to_use.signature.apply_argn(idx, dt)")
            args_to_use.append(arg_to_use)

        fc = P.FunctionCall(func_to_use, tuple(args_to_use))
        return fc

    def generate(self, context: Context, name: str) -> P.Function | None:
        # clears the frontier
        context.statements.clear()
        context.input_args.clear()

        # NOTE: #2 - Sample the argument types and return type
        input_types = tuple(it for it in self._arc_types for _ in range(3))
        return_type = self._sample_dtype()

        arg_list = tuple(
            P.Arg(f"a{idx}", dtype, idx) for idx, dtype in enumerate(input_types)
        )
        context.input_args.extend(arg_list)

        if not (
            fc := self._generate_funccall_from_rettype(context, return_type, depth=0)
        ):
            return None

        stmt = self._create_add_statement(context, fc)
        arg_list = tuple(
            filter(
                lambda a: any(
                    a.caller_idx == arg.caller_idx
                    for stmt in context.statements
                    for arg in stmt.rhs.args
                    if isinstance(arg, P.Arg)
                ),
                arg_list,
            )
        )
        ret_stmt = P.ReturnStmt(stmt)
        return P.Function(
            name, arg_list, return_type, tuple(context.statements), ret_stmt
        )
