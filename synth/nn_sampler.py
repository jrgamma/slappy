from typing import Iterable, TypeVar


import torch
from torch import nn
from torch.nn import functional as F

from synth import primitives as P
from synth.generator import ArgTypeSources
from synth.generator import Sampler, SamplingTypes


SamplingDataTypes = TypeVar(
    "SamplingDataTypes",
    str,
    P.DataType,
    ArgTypeSources,
    P.Arg,
    P.AssgnStmt,
    P.ConstantDataType,
)


def repeat_concat_tensor(
    t1: torch.Tensor, t2: torch.Tensor, d1: int = 0, d2: int = -1
) -> torch.Tensor:
    reps = t2.size(d1)
    t3 = torch.repeat_interleave(t1, reps, d1)
    t4 = torch.cat([t3, t2], dim=d2)
    return t4


class DataEmbedding[_T](nn.Module):

    PAD_KEY = 0xDEADBEEF

    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=vocab_size - 1, max_norm=2.0
        )
        self._data_idx_map: dict[int, int] = {
            self.PAD_KEY: vocab_size - 1,
        }

    @property
    def pad_index(self) -> int:
        return self._vocab_size - 1

    def hash_key(self, value: _T | None) -> int:
        if value is None:
            return self.PAD_KEY
        return hash(value)

    def prepare(self, lst: Iterable[_T | None]) -> None:
        for item in lst:
            key = self.hash_key(item)
            if key in self._data_idx_map:
                continue
            idx = len(self._data_idx_map)
            if idx >= self._vocab_size:
                raise ValueError(
                    f"{self.__class__.__name__}>> Exceeded vocab size, item: {item}"
                )
            self._data_idx_map[key] = idx

    def forward(self, data: Iterable[_T | None]) -> torch.Tensor:
        with torch.no_grad():
            self.prepare(data)
        indices = torch.tensor(
            list(map(lambda it: self._data_idx_map[self.hash_key(it)], data)),
            dtype=torch.long,
        )
        return self.embedding(indices)


class StrDataEmbedding(DataEmbedding[str]):
    pass


class DataTypeEmbedding(DataEmbedding[P.DataType]):
    pass


class ConstantEmbedding(DataEmbedding[P.ConstantDataType]):
    pass


class ImportedFunctionEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        max_classes: int,
        max_functions: int,
        class_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self._class_dim = int(class_fraction * embed_dim)
        self._function_dim = embed_dim - self._class_dim
        self.class_embedding = StrDataEmbedding(self._class_dim, max_classes)
        self.function_embedding = StrDataEmbedding(self._function_dim, max_functions)

    def _split_data(
        self, lst: Iterable[str]
    ) -> tuple[list[str | None], list[str | None]]:
        list_class: list[str | None] = []
        list_funcs: list[str | None] = []
        for item in lst:
            if item:
                a, b = item.split(".")
                list_class.append(a)
                list_funcs.append(b)
            else:
                list_class.append(None)
                list_funcs.append(None)

        return list_class, list_funcs

    def prepare(self, lst: Iterable[str]) -> None:
        list_class, list_funcs = self._split_data(lst)
        self.class_embedding.prepare(list_class)
        self.function_embedding.prepare(list_funcs)

    def forward(self, data: Iterable[str]) -> torch.Tensor:
        list_class, list_funcs = self._split_data(data)
        emb_class = self.class_embedding(list_class)
        emb_funcs = self.function_embedding(list_funcs)
        return torch.cat([emb_class, emb_funcs], dim=-1)


class ArgEmbedding(nn.Module):

    def __init__(
        self,
        data_type_embedding: DataTypeEmbedding,
        constant_embedding: ConstantEmbedding,
        idx_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.dt_embed = data_type_embedding
        self.constant_embed = constant_embedding
        self._idx_weight = idx_weight
        self._dt_weight = 1 - idx_weight

    def forward(self, data: Iterable[P.Arg | None]) -> torch.Tensor:
        data_types = [item.dtype if item else None for item in data]
        indices = [item.caller_idx if item else None for item in data]
        x1 = self.dt_embed(data_types)
        x2 = self.constant_embed(indices)
        return (x1 * self._dt_weight + x2 * self._idx_weight) / (
            self._idx_weight + self._dt_weight
        )


class StatementEmbedding(nn.Module):

    def __init__(
        self,
        data_type_embedding: DataTypeEmbedding,
        arg_embedding: ArgEmbedding,
        constant_embedding: ConstantEmbedding,
        function_embedding: ImportedFunctionEmbedding,
        args_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dt_embed = data_type_embedding
        self.arg_embed = arg_embedding
        self.const_embed = constant_embedding
        self.func_embed = function_embedding
        self._args_weight = args_weight
        self._rtype_weight = 1 - args_weight

    def flatten_items(self, data: Iterable[P.AssgnStmt]):
        rtypes = [item.dtype for item in data]
        args = torch.stack(
            [
                self.arg_embed(
                    [arg if isinstance(arg, P.Arg) else None for arg in item.rhs.args]
                ).sum(0)
                for item in data
            ],
            dim=0,
        )
        stmts = torch.stack(
            [
                self.dt_embed(
                    [
                        arg.dtype if isinstance(arg, P.AssgnStmt) else None
                        for arg in item.rhs.args
                    ]
                ).sum(0)
                for item in data
            ],
            dim=0,
        )
        consts = torch.stack(
            [
                self.const_embed(
                    [
                        arg.value if isinstance(arg, P.Constant) else None
                        for arg in item.rhs.args
                    ]
                ).sum(0)
                for item in data
            ],
            dim=0,
        )
        funcs = torch.stack(
            [
                self.func_embed(
                    [
                        arg.name if isinstance(arg, P._Function) else None
                        for arg in item.rhs.args
                    ]
                ).sum(0)
                for item in data
            ],
            dim=0,
        )
        nargs = torch.tensor([len(item.rhs.args) for item in data]).unsqueeze(-1)
        return rtypes, (args + stmts + consts + funcs) / nargs

    def forward(self, data: Iterable[P.AssgnStmt]) -> torch.Tensor:
        rtypes, argss = self.flatten_items(data)
        return (
            self.dt_embed(rtypes) * self._rtype_weight + argss * self._args_weight
        ) / (self._rtype_weight + self._args_weight)


class SamplingModel(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        match_dim: int,
        embeddings: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self._scale = 2
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict(embeddings)
        self.gru = nn.GRU(embed_dim, hidden_dim, dropout=0.25)
        self.linear_gru = nn.Linear(hidden_dim, match_dim, bias=False)
        self.linear_embed = nn.Linear(embed_dim, match_dim, bias=False)
        self.register_buffer("hidden_state", self._scale * torch.randn((1, hidden_dim)))
        self.register_buffer(
            "previous_selection", self._scale * torch.zeros(1, embed_dim)
        )
        self.scores: list[torch.Tensor] = []

    def reset_state(self, seed: int):
        torch.manual_seed(seed % 0xDEADBEEF)
        self.hidden_state = self._scale * torch.randn((1, self.hidden_dim))
        self.previous_selection = self._scale * torch.zeros((1, self.embed_dim))
        self.scores.clear()

    def forward(
        self, choices: list[SamplingDataTypes], stype: SamplingTypes
    ) -> SamplingDataTypes:
        embeds = self.embeddings[stype.name](choices)
        _, gru_out = self.gru(self.previous_selection, self.hidden_state)

        # compute score
        queries = self.linear_embed(embeds)
        key = self.linear_gru(gru_out)
        scores = F.softmax(key @ queries.transpose(-2, -1), dim=-1)
        index = torch.argmax(scores, dim=-1).detach().item()

        # update
        self.previous_selection = torch.cat(
            (self.previous_selection, embeds[index : (index + 1)]), dim=0
        )
        self.hidden_state = gru_out
        self.scores.append(scores[..., int(index)])

        return choices[int(index)]


class NeuralSampler(Sampler):

    def __init__(
        self, hidden_dim: int, embed_dim: int, match_dim: int, seed: int | None = None
    ) -> None:
        super().__init__(seed)
        torch.manual_seed(self._seed % 0xDEADBEEF)
        data_type_embedding = DataTypeEmbedding(embed_dim, 200)
        function_embedding = ImportedFunctionEmbedding(embed_dim, 100, 300)
        arg_source_embedding = DataEmbedding[ArgTypeSources](
            embed_dim, len(list(ArgTypeSources)) + 2
        )
        constant_embedding = ConstantEmbedding(embed_dim, 100)
        arg_apply_embedding = ArgEmbedding(data_type_embedding, constant_embedding)
        statements_embedding = StatementEmbedding(
            data_type_embedding,
            arg_apply_embedding,
            constant_embedding,
            function_embedding,
        )
        embeddings = {
            SamplingTypes.io_data_types.name: data_type_embedding,
            SamplingTypes.imported_function.name: function_embedding,
            SamplingTypes.arg_source.name: arg_source_embedding,
            SamplingTypes.constant.name: constant_embedding,
            SamplingTypes.arg_to_apply.name: arg_apply_embedding,
            SamplingTypes.statements.name: statements_embedding,
        }
        self.model = SamplingModel(hidden_dim, embed_dim, match_dim, embeddings)

    def sample(
        self, choices: list[SamplingDataTypes], stype: SamplingTypes
    ) -> SamplingDataTypes:
        out = self.model(choices, stype)
        # print(f"  nn sampled output = {out}")
        return out

    def reset_state(self):
        self._seed += 1
        self.model.reset_state(self._seed)
