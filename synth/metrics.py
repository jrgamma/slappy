from __future__ import annotations

import abc
import difflib
import itertools
import random
import json
import zlib

import numpy as np

from synth import primitives as P

class UnsupportedTypes(Exception):
    pass


class RootSimilarity:

    def __init__(self):
        self.children: list[ISimilarity] = []

    def add_child(self, child: ISimilarity):
        child.set_root(self)
        self.children.append(child)

    def compare(self, first: object, second: object) -> float:
        if first == second:
            return 1.0
        try:
            for child in self.children:
                out = child.compare(first, second)
                if isinstance(out, float):
                    return out
        except UnsupportedTypes:
            pass

        return 0.0


class ISimilarity[_T1, _T2]:

    def __init__(self):
        self.root: RootSimilarity

    def set_root(self, root: RootSimilarity):
        self.root = root

    @abc.abstractmethod
    def _compare(self, first: _T1, second: _T2) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def check_first(self, item: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def check_second(self, item: object) -> bool:
        raise NotImplementedError

    def compare(self, first: object, second: object) -> float:
        """Computes the similarity score."""
        if self.check_first(first) and self.check_second(second):
            return self._compare(first, second)  # type: ignore
        if self.check_second(first) and self.check_first(second):
            return self._compare(second, first)  # type: ignore
        raise UnsupportedTypes


def gaussian_distance(first: int | float, second: int | float) -> float:
    return np.exp(-((first - second) ** 2)).item()


class IntIntSimilarity(ISimilarity[int, int]):

    def check_first(self, item: object) -> bool:
        return isinstance(item, int)

    def check_second(self, item: object) -> bool:
        return isinstance(item, int)

    def _compare(self, first: int, second: int) -> float:
        return gaussian_distance(first, second)


class IntPairSimilarity(ISimilarity[tuple[int, int], tuple[int, int]]):

    @staticmethod
    def check(item: object) -> bool:
        return (
            isinstance(item, tuple) and len(item) == 2 and all(isinstance(it, int) for it in item)
        )

    def check_first(self, item: object) -> bool:
        return self.check(item)

    def check_second(self, item: object) -> bool:
        return self.check(item)

    def _compare(self, first: tuple[int, int], second: tuple[int, int]) -> float:
        return 0.5 * (
            gaussian_distance(first[0], second[0]) + gaussian_distance(first[1], second[1])
        )


class IntListSimilarity(ISimilarity[tuple[int, ...], tuple[int, ...]]):

    @staticmethod
    def check(item: object) -> bool:
        if not isinstance(item, tuple):
            return False
        if len(item):
            return isinstance(next(iter(item)), int)
        return True

    def check_first(self, item: object) -> bool:
        return self.check(item)

    def check_second(self, item: object) -> bool:
        return self.check(item)

    def _compare(self, first: tuple[int, ...], second: tuple[int, ...]) -> float:
        sims: list[float] = []

        l1 = len(first)
        l2 = len(second)
        sims.append(gaussian_distance(l1, l2))

        max1 = max(first)
        max2 = max(second)
        sims.append(gaussian_distance(max1, max2))

        min1 = min(first)
        min2 = min(second)
        sims.append(gaussian_distance(min1, min2))

        mean1 = np.mean(first).item()
        mean2 = np.mean(second).item()
        sims.append(gaussian_distance(mean1, mean2))

        std1 = np.std(first).item()
        std2 = np.std(second).item()
        sims.append(gaussian_distance(std1, std2))

        sm = difflib.SequenceMatcher(None, first, second)
        sim_seq = sm.find_longest_match().size / max(l1, l2)
        sims.append(sim_seq)

        s1 = frozenset(first)
        s2 = frozenset(second)
        sims.append(self.root.compare(s1, s2))

        nelem1 = len(set(first))
        nelem2 = len(set(second))
        sims.append(gaussian_distance(nelem1, nelem2))

        return np.mean(sims).item()


class IntSetSimilarity(ISimilarity[frozenset[int], frozenset[int]]):

    @staticmethod
    def check(item: object) -> bool:
        if not isinstance(item, frozenset):
            return False
        if len(item):
            return isinstance(next(iter(item)), int)
        return True

    def check_first(self, item: object) -> bool:
        return self.check(item)

    def check_second(self, item: object) -> bool:
        return self.check(item)

    def _compare(self, first: frozenset[int], second: frozenset[int]) -> float:
        sims: list[float] = []

        l1 = len(first)
        l2 = len(second)
        sims.append(gaussian_distance(l1, l2))

        max1 = max(first)
        max2 = max(second)
        sims.append(gaussian_distance(max1, max2))

        min1 = min(first)
        min2 = min(second)
        sims.append(gaussian_distance(min1, min2))

        mean1 = np.mean(list(first)).item()
        mean2 = np.mean(list(second)).item()
        sims.append(gaussian_distance(mean1, mean2))

        std1 = np.std(list(first)).item()
        std2 = np.std(list(second)).item()
        sims.append(gaussian_distance(std1, std2))

        intersection = first.intersection(second)
        sim_seq = len(intersection) / max(l1, l2)
        sims.append(sim_seq)

        return np.mean(sims).item()


class IntSetListSimilarity(ISimilarity[frozenset[int], tuple[int, ...]]):

    def check_first(self, item: object) -> bool:
        return IntSetSimilarity.check(item)

    def check_second(self, item: object) -> bool:
        return IntListSimilarity.check(item)

    def _compare(self, first: frozenset[int], second: tuple[int, ...]) -> float:
        return max(
            self.root.compare(first, frozenset(second)),
            self.root.compare(tuple(first), second),
        )


class GenericContainerSimilarity(ISimilarity[tuple | frozenset, tuple | frozenset]):

    def check_first(self, item: object) -> bool:
        return isinstance(item, tuple | frozenset)

    def check_second(self, item: object) -> bool:
        return isinstance(item, tuple | frozenset)

    def _compare(self, first: tuple | frozenset, second: tuple | frozenset) -> float:

        l1 = len(first)
        l2 = len(second)
        sim_len = gaussian_distance(l1, l2)

        sims: list[float] = []
        for a, b in zip(first, second):
            sims.append(self.root.compare(a, b))

        return 0.5 * (sim_len + np.mean(sims).item())


def set_default(obj: object):
    if isinstance(obj, frozenset):
        return list(obj)
    raise TypeError(str(obj))


class CompressionDistance:

    def _compress(self, obj: object) -> int:
        if isinstance(obj, (int, float, str)):
            obj_bytes = str(obj).encode("utf-8")
        else:
            obj_bytes = json.dumps(obj, default=set_default).encode("utf-8")
        return len(zlib.compress(obj_bytes))

    def _ncd(self, x: object, y: object) -> float:
        """Compute Normalized Compression Distance between x and y."""
        Cx = self._compress(x)
        Cy = self._compress(y)
        Cxy = self._compress((x, y))  # serialize as a tuple
        return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

    def compute_sim(self, x: object, y: object) -> float:
        return 1.0 - self._ncd(x, y)


class Metric:

    def __init__(self):
        self.cd = CompressionDistance()

        self.sims = RootSimilarity()
        self.sims.add_child(IntIntSimilarity())
        self.sims.add_child(IntPairSimilarity())
        self.sims.add_child(IntListSimilarity())
        self.sims.add_child(IntSetSimilarity())
        self.sims.add_child(IntSetListSimilarity())
        self.sims.add_child(GenericContainerSimilarity())

    def _compute_sim_single(self, x: object, y: object) -> float:
        s1 = self.sims.compare(x, y)
        s2 = self.cd.compute_sim(x, y)
        return max(s1, s2)

    def compute_sim(self, x: list[tuple[object, ...]], y: list[object]) -> float:
        io_sim = np.mean(
            [max(self._compute_sim_single(a, b) for a in ass) for ass, b in zip(x, y)]
        ).item()

        # select len pairs of outputs to compute similarity between them
        all_pairs = list(itertools.combinations(y, 2))
        selected_pairs = random.sample(all_pairs, len(y))
        intra_sim = np.mean([self._compute_sim_single(a, b) for a, b in selected_pairs]).item()

        return max(io_sim, intra_sim)

    def score(
        self, func: P.Function, inputs: list[tuple[object, ...]], outputs: list[object]
    ) -> float:
        score_length = np.tanh(len(func.body) / np.exp(1)).item()
        score_sim = 1.0 - self.compute_sim(inputs, outputs)

        return 0.5 * score_length + 0.5 * score_sim
