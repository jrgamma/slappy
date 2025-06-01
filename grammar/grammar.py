"""Grammar of primitives."""

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")

_P1 = TypeVar("_P1")
_P2 = TypeVar("_P2")
_P3 = TypeVar("_P3")


class _BaseOps[_T]:
    """Base operations for every type."""

    @classmethod
    def identity(cls, x: _T) -> _T:
        return x


class _ComparableProperties[_T]:
    """Items that support some comparisons."""

    @classmethod
    def equality(cls, a: _T, b: _T) -> bool:
        return a == b


class IntOps(_BaseOps[int]):
    """Arithmetic operations.

    Function signature:
    ```python
        (int, ...) -> int
    ```
    """

    @classmethod
    def add(cls, a: int, b: int) -> int:
        return a + b

    @classmethod
    def subtract(cls, a: int, b: int) -> int:
        return a - b

    @classmethod
    def multiply(cls, a: int, b: int) -> int:
        return a * b

    @classmethod
    def divide(cls, a: int, b: int) -> int:
        return a // b

    @classmethod
    def invert(cls, x: int) -> int:
        return -x

    @classmethod
    def double(cls, x: int) -> int:
        return 2 * x

    @classmethod
    def halve(cls, x: int) -> int:
        return x // 2

    @classmethod
    def increment(cls, x: int) -> int:
        return x + 1

    @classmethod
    def decrement(cls, x: int) -> int:
        return x - 1

    @classmethod
    def crement(cls, x: int) -> int:
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)

    @classmethod
    def sign(cls, x: int) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)


class IntProperties(_ComparableProperties[int]):
    """int properties: is-a relationship.

    Signature:
    ```python
        (int, ...) -> bool
    ```
    """

    @classmethod
    def even(cls, x: int) -> bool:
        return (x % 2) == 0

    @classmethod
    def greater(cls, a: int, b: int) -> bool:
        return a > b

    @classmethod
    def positive(cls, x: int) -> bool:
        return x > 0


class BoolOps(_BaseOps[bool]):
    """Boolean operations.

    Signature:
    ```python
        (Booelan, ...) -> bool
    ```
    """

    @classmethod
    def flip(cls, x: bool) -> bool:
        return not x

    @classmethod
    def both(cls, a: bool, b: bool) -> bool:
        return a and b

    @classmethod
    def either(cls, a: bool, b: bool) -> bool:
        return a or b


class SetOps[_T](_BaseOps[frozenset[_T]]):
    """Set operations.

    Signature:
    ```python
        (frozenset, ...) -> frozenset
    ```
    """

    @classmethod
    def combine(cls, a: frozenset[_T], b: frozenset[_T]) -> frozenset[_T]:
        return frozenset((*a, *b))

    @classmethod
    def intersection(cls, a: frozenset[_T], b: frozenset[_T]) -> frozenset[_T]:
        return a & b

    @classmethod
    def difference(cls, a: frozenset[_T], b: frozenset[_T]) -> frozenset[_T]:
        return a - b

    @classmethod
    def sfilter(cls, x: frozenset[_T], f: Callable[[_T], bool]) -> frozenset[_T]:
        return frozenset(e for e in x if f(e))

    @classmethod
    def insert(cls, x: frozenset[_T], value: _T) -> frozenset[_T]:
        return x.union(frozenset({value}))

    @classmethod
    def remove(cls, x: frozenset[_T], value: _T) -> frozenset[_T]:
        return SetOps[_T].sfilter(x, lambda e: e == value)

    @classmethod
    def apply(cls, x: frozenset[_T], func: Callable[[_T], _R]) -> frozenset[_R]:
        return frozenset(func(e) for e in x)


class SetExtractionOps[_T]:
    """Set operations that extracts elements.

    Signature:
    ```python
        (frozenset[_T], ...) -> _T
    ```
    """

    @classmethod
    def extract(cls, x: frozenset[_T], f: Callable[[_T], bool]) -> _T:
        return next(e for e in x if f(e))

    @classmethod
    def anyitem(cls, x: frozenset[_T]) -> _T:
        return next(iter(x))

    @classmethod
    def anyother(cls, x: frozenset[_T], value: _T) -> _T:
        return SetExtractionOps[_T].anyitem(SetOps[_T].remove(x, value))


class NestedSetOps[_T](SetOps[frozenset[_T]]):
    """Nested set operations.

    Signature:
    ```python
        (frozenset[frozenset[_T]], ...) -> frozenset[_T]
    ```
    """

    @classmethod
    def flatten(cls, sets: frozenset[frozenset[_T]]) -> frozenset[_T]:
        return frozenset(e for s in sets for e in s)

    @classmethod
    def mfilter(cls, sets: frozenset[frozenset[_T]], func: Callable[[frozenset[_T]], bool]) -> frozenset[_T]:
        return cls.flatten(cls.sfilter(sets, func))


class SetProperties[_T](_ComparableProperties[frozenset[_T]]):
    """Set properties.

    Signature:
    ```python
        (FrozenSet, ...) -> bool | int
    ```
    """

    @classmethod
    def contained(cls, x: frozenset[_T], a: _T) -> bool:
        return a in x

    @classmethod
    def size(cls, x: frozenset[_T]) -> int:
        return len(x)

    @classmethod
    def maximum(cls, x: frozenset[int]) -> int:
        return max(x, default=0)

    @classmethod
    def minimum(cls, x: frozenset[int]) -> int:
        return min(x, default=0)

    @classmethod
    def argmax(cls, x: frozenset[_T], f: Callable[[_T], int]) -> _T:
        return max(x, key=f)

    @classmethod
    def argmin(cls, x: frozenset[_T], f: Callable[[_T], int]) -> _T:
        return min(x, key=f)

    @classmethod
    def valmax(cls, x: frozenset[_T], f: Callable[[_T], int]) -> int:
        return f(SetProperties[_T].argmax(x, f))

    @classmethod
    def valmin(cls, x: frozenset[_T], f: Callable[[_T], int]) -> int:
        return f(SetProperties[_T].argmin(x, f))


class SetGenerators[_T]:
    """Generate sets.

    Signature:
    ```python
        (_T, ...) -> frozenset[_T]
    ```
    """

    @classmethod
    def initset(cls, x: _T) -> frozenset[_T]:
        return frozenset({x})


class SetGeneratorsPair[_S, _T]:

    @classmethod
    def productset(cls, a: frozenset[_S], b: frozenset[_T]) -> frozenset[tuple[_S, _T]]:
        return frozenset({(i, j) for j in b for i in a})

    @classmethod
    def productlist(cls, a: tuple[_S, ...], b: tuple[_T, ...]) -> frozenset[tuple[_S, _T]]:
        return frozenset({(i, j) for j in b for i in a})


class ListOps[_T](_BaseOps[tuple[_T, ...]]):
    """Immutable list operations.

    Signature:
    ```python
        (tuple[_T, ...], ...) -> tuple[_T, ...]
    ```
    """

    @classmethod
    def combine(cls, a: tuple[_T, ...], b: tuple[_T, ...]) -> tuple[_T, ...]:
        return tuple((*a, *b))

    @classmethod
    def dedupe(cls, x: tuple[_T, ...]) -> tuple[_T, ...]:
        return tuple(e for i, e in enumerate(x) if x.index(e) == i)

    @classmethod
    def order(cls, x: tuple[_T, ...], f: Callable[[_T], int]) -> tuple[_T, ...]:
        return tuple(sorted(x, key=f))

    @classmethod
    def sfilter(cls, x: tuple[_T, ...], f: Callable[[_T], bool]) -> tuple[_T, ...]:
        return tuple(e for e in x if f(e))

    @classmethod
    def append(cls, x: tuple[_T, ...], value: _T) -> tuple[_T, ...]:
        return x + (value,)

    @classmethod
    def prepend(cls, x: tuple[_T, ...], value: _T) -> tuple[_T, ...]:
        return (value,) + x

    @classmethod
    def remove(cls, x: tuple[_T, ...], value: _T) -> tuple[_T, ...]:
        return cls.sfilter(x, lambda e: e == value)

    @classmethod
    def apply(cls, x: tuple[_T, ...], func: Callable[[_T], _R]) -> tuple[_R, ...]:
        return tuple(func(e) for e in x)


class ListExtractionOps[_T]:
    """List operations that extracts elements.

    Signature:
    ```python
        (tuple[_T, ...], ...) -> _T
    ```
    """

    @classmethod
    def extract(cls, x: tuple[_T, ...], f: Callable[[_T], bool]) -> _T:
        return next(e for e in x if f(e))

    @classmethod
    def first(cls, x: tuple[_T, ...]) -> _T:
        return next(iter(x))

    @classmethod
    def last(cls, x: tuple[_T, ...]) -> _T:
        return max(enumerate(x))[1]

    @classmethod
    def firstother(cls, x: tuple[_T, ...], value: _T) -> _T:
        return cls.first(ListOps[_T].remove(x, value))

    @classmethod
    def at(cls, x: tuple[_T, ...], n: int) -> _T:
        return x[n]


class NestedListOps[_T](ListOps[tuple[_T, ...]]):
    """Nested list operations.

    Signature:
    ```python
        (tuple[tuple[_T, ...], ...], ...) -> tuple[_T, ...]
    """

    @classmethod
    def flatten(cls, tuples: tuple[tuple[_T, ...], ...]) -> tuple[_T, ...]:
        return tuple(e for t in tuples for e in t)

    @classmethod
    def mfilter(cls, tuples: tuple[tuple[_T, ...], ...], func: Callable[[tuple[_T, ...]], bool]) -> tuple[_T, ...]:
        return cls.flatten(cls.sfilter(tuples, func))

    @classmethod
    def mapply(
        cls,
        tuples: tuple[tuple[_T, ...], ...],
        func: Callable[[tuple[_T, ...]], tuple[_R, ...]],
    ) -> tuple[_R, ...]:
        return NestedListOps[_R].flatten(cls.apply(tuples, func))


class ListProperties[_T](_ComparableProperties[tuple[_T, ...]]):
    """List properties.

    Signature:
    ```python
        (tuple, ...) -> bool | int
    ```
    """

    @classmethod
    def contained(cls, x: tuple[_T, ...], a: _T) -> bool:
        return a in x

    @classmethod
    def size(cls, x: tuple[_T, ...]) -> int:
        return len(x)

    @classmethod
    def maximum(cls, x: tuple[int, ...]) -> int:
        return max(x, default=0)

    @classmethod
    def minimum(cls, x: tuple[int, ...]) -> int:
        return min(x, default=0)

    @classmethod
    def argmax(cls, x: tuple[_T, ...], f: Callable[[_T], int]) -> _T:
        return max(x, key=f)

    @classmethod
    def argmin(cls, x: tuple[_T, ...], f: Callable[[_T], int]) -> _T:
        return min(x, key=f)

    @classmethod
    def valmax(cls, x: tuple[_T, ...], f: Callable[[_T], int]) -> int:
        return f(ListProperties[_T].argmax(x, f))

    @classmethod
    def valmin(cls, x: tuple[_T, ...], f: Callable[[_T], int]) -> int:
        return f(ListProperties[_T].argmin(x, f))

    @classmethod
    def mostcommon(cls, x: tuple[_T, ...]) -> _T:
        return max(set(x), key=x.count)

    @classmethod
    def leastcommon(cls, x: tuple[_T, ...]) -> _T:
        return min(set(x), key=x.count)


class ListGenerators[_T]:
    """Generate list.

    Signature:
    ```python
        (_T, ...) -> tuple[_T, ...]
    ```
    """

    @classmethod
    def repeat(cls, item: _T, rep: int) -> tuple[_T, ...]:
        return tuple(item for _ in range(rep))

    @classmethod
    def interval(cls, start: int, stop: int, step: int) -> tuple[int, ...]:
        return tuple(range(start, stop, step))


class ListGeneratorsPair[_S, _T]:

    @classmethod
    def pair(cls, a: tuple[_S, ...], b: tuple[_T, ...]) -> tuple[tuple[_S, _T], ...]:
        return tuple(zip(a, b))


class SetToListOps[_T]:
    """Set operations that result in lists.

    Signature:
    ```python
        (frozenset[_T], ...) -> tuple[_T, ...]
    ```
    """

    @classmethod
    def tolist(cls, x: frozenset[_T]) -> tuple[_T, ...]:
        return tuple(x)

    @classmethod
    def order(cls, x: frozenset[_T], f: Callable[[_T], int]) -> tuple[_T, ...]:
        return tuple(sorted(x, key=f))


class ListToSetOps[_T]:

    @classmethod
    def toset(cls, x: tuple[_T, ...]) -> frozenset[_T]:
        return frozenset(x)


class TupleGenerators[_S, _T]:

    @classmethod
    def astuple(cls, a: _S, b: _T) -> tuple[_S, _T]:
        return (a, b)


class TupleOps[_S, _T]:

    @classmethod
    def first(cls, tup: tuple[_S, _T]) -> _S:
        return tup[0]

    @classmethod
    def second(cls, tup: tuple[_S, _T]) -> _T:
        return tup[1]


class ControlFlow:

    @classmethod
    def branch(cls, condition: bool, a: _T, b: _T) -> _T:
        return a if condition else b


class Functional:

    @classmethod
    def compose(cls, outer: Callable[[_P2], _R], inner: Callable[[_P1], _P2]) -> Callable[[_P1], _R]:
        return lambda x: outer(inner(x))

    @classmethod
    def chain(cls, h: Callable[[_P3], _R], g: Callable[[_P2], _P3], f: Callable[[_P1], _P2]) -> Callable[[_P1], _R]:
        return lambda x: h(g(f(x)))

    @classmethod
    def matcher(cls, func: Callable[[_T], _R], target: _R) -> Callable[[_T], bool]:
        return lambda x: func(x) == target

    @classmethod
    def bind(cls, func: Callable[[_T], _R], fixed: _T) -> Callable[[], _R]:
        return lambda: func(fixed)

    @classmethod
    def rbind2(cls, func: Callable[[_P1, _T], _R], fixed: _T) -> Callable[[_P1], _R]:
        return lambda x: func(x, fixed)

    @classmethod
    def rbind3(cls, func: Callable[[_P1, _P2, _T], _R], fixed: _T) -> Callable[[_P1, _P2], _R]:
        return lambda x1, x2: func(x1, x2, fixed)

    @classmethod
    def lbind2(cls, func: Callable[[_T, _P1], _R], fixed: _T) -> Callable[[_P1], _R]:
        return lambda x: func(fixed, x)

    @classmethod
    def lbind3(cls, func: Callable[[_T, _P1, _P2], _R], fixed: _T) -> Callable[[_P1, _P2], _R]:
        return lambda x1, x2: func(fixed, x1, x2)

    @classmethod
    def power(cls, func: Callable[[_T], _T], n: int) -> Callable[[_T], _T]:
        if n <= 1:
            return func
        return Functional.compose(func, Functional.power(func, n - 1))

    @classmethod
    def fork(
        cls,
        outer: Callable[[_P1, _P2], _R],
        a: Callable[[_T], _P1],
        b: Callable[[_T], _P2],
    ) -> Callable[[_T], _R]:
        return lambda x: outer(a(x), b(x))

    @classmethod
    def setapply(cls, funcs: frozenset[Callable[[_T], _R]], value: _T) -> frozenset[_R]:
        return frozenset(func(value) for func in funcs)

    @classmethod
    def listapply(cls, funcs: tuple[Callable[[_T], _R], ...], value: _T) -> tuple[_R, ...]:
        return tuple(func(value) for func in funcs)


class FunctionalPair:

    @classmethod
    def papply(cls, func: Callable[[_P1, _P2], _R], a: tuple[_P1, ...], b: tuple[_P2, ...]) -> tuple[_R, ...]:
        return tuple(func(i, j) for i, j in zip(a, b))

    @classmethod
    def mpapply(
        cls,
        func: Callable[[_P1, _P2], tuple[_R, ...]],
        a: tuple[_P1, ...],
        b: tuple[_P2, ...],
    ) -> tuple[_R, ...]:
        return NestedListOps[_R].flatten(FunctionalPair.papply(func, a, b))

    @classmethod
    def prapplylist(cls, func: Callable[[_P1, _P2], _R], a: tuple[_P1, ...], b: tuple[_P2, ...]) -> frozenset[_R]:
        return frozenset({func(i, j) for j in b for i in a})

    @classmethod
    def prapplyset(cls, func: Callable[[_P1, _P2], _R], a: frozenset[_P1], b: frozenset[_P2]) -> frozenset[_R]:
        return frozenset({func(i, j) for j in b for i in a})
