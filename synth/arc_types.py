from __future__ import annotations
import enum

from synth import primitives as P

_Tuple = P.ContainerType.Tup
_Set = P.ContainerType.Set


class ArcTypes(enum.Enum):
    Bool = P.ScalarType.Boolean
    Int = P.ScalarType.Int

    IntPair = P.IContainer((Int, Int))
    Cell = P.IContainer((Int, IntPair))

    IntList = P.HContainer(_Tuple, Int)
    Grid = P.HContainer(_Tuple, IntList)

    IntSet = P.HContainer(_Set, Int)
    Indices = P.HContainer(_Set, IntPair)
    Object = P.HContainer(_Set, Cell)

    IndicesSet = P.HContainer(_Set, Indices)
    Objects = P.HContainer(_Set, Object)


class ArcBoolConstant(enum.Enum):
    false = False
    true = True


class ArcIntConstant(enum.Enum):
    neg_one = -1
    neg_two = -2
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10


class ArcIndexConstant(enum.Enum):
    right = (1, 0)
    top = (0, 1)
    left = (-1, 0)
    bottom = (0, -1)
    origin = (0, 0)
    northeast = (1, 1)
    southwest = (-1, -1)
    northwest = (-1, 1)
    southeast = (1, -1)
    zero_by_two = (0, 2)
    two_by_zero = (2, 0)
    two_by_two = (2, 2)
    three_by_three = (3, 3)


ARC_CONSTANTS: dict[
    P.DataType, type[ArcBoolConstant] | type[ArcIntConstant] | type[ArcIndexConstant]
] = {
    ArcTypes.Bool.value: ArcBoolConstant,
    ArcTypes.Int.value: ArcIntConstant,
    ArcTypes.IntPair.value: ArcIndexConstant,
}
