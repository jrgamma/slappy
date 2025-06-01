IntPair = tuple[int, int]
Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, IntPair]

Object = frozenset[Cell]
Objects = frozenset[Object]
Indices = frozenset[IntPair]
IndicesSet = frozenset[Indices]
