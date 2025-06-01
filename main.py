import argparse
import ast
import datetime
import json
import importlib
import types
import uuid
from typing import Iterable, Literal, TypeAlias

from synth import data_generator as DG
from synth import generator as Gen
from synth import primitives as P
from synth import nn_sampler as NN
from synth.metrics import Metric
from grammar import grammar as G

import numpy as np
import torch
from torch import optim


ResultType: TypeAlias = dict[str, dict[Literal["inputs", "outputs"], list]]

GEN_DIR_NAME = "out/generated"
MODEL_DIR_NAME = "out/models"
LOG_DIR_NAME = "out/logs"
MAGIC_NUMBER = 27374335
MODULE_ALIAS = "G"


def compile_function(function: P.Function) -> types.FunctionType:
    node = ast.parse(function.to_definition())
    node = ast.fix_missing_locations(node)
    code = compile(node, "", mode="exec")
    namespace = {MODULE_ALIAS: G}
    exec(code, namespace)
    return namespace.get(function.name)


def generate_pairs(
    dg: DG.MainDataGenerator,
    func_exe: types.FunctionType,
    func_item: P.Function,
    num_samples: int,
):
    inputs = []
    outputs = []
    for _ in range(num_samples):
        ctx = DG.DataGenContext()
        current_inputs = tuple(dg.generate(ctx, arg.dtype) for arg in func_item.args)
        current_outputs = func_exe(*current_inputs)
        inputs.append(current_inputs)
        outputs.append(current_outputs)

    return inputs, outputs


def generate_functions(
    sampler: Gen.Sampler, num_functions: int, grammar: Gen.Grammar, batch_no: int
) -> dict[str, P.Function]:
    functions: dict[str, P.Function] = {}
    function = None
    for fcount in range(1, num_functions + 1):
        seed = ((batch_no << 10) | (fcount << 5)) ^ MAGIC_NUMBER
        sampler.reseed(seed)
        print(f"Iteration: {fcount}, seed: {hex(seed)}")
        while True:
            try:
                fid = str(uuid.uuid4()).split("-")[0]
                ctx = Gen.Context(grammar=grammar)
                gen = Gen.FunctionGenerator(sampler)
                function = gen.generate(ctx, f"func_{fid}")
            except Gen.TooMuchNesting:
                pass
            except Exception as e:
                print(f"[ERROR]: {e.__class__.__name__}, {e!s}")
            else:
                if function and function.args:
                    break

            sampler.reset_state()

        if function:
            functions[fid] = function
            print("Generated!!!")
            print()

    print(len(functions))
    return functions


def write_functions(
    filename: str, func_list: Iterable[P.Function], alias: str = MODULE_ALIAS
):
    with open(filename, "w") as f:
        f.write(f"from grammar import grammar as {alias}\n\n\n")
        for func in func_list:
            f.write(func.to_definition())
            f.write("\n\n\n")


def generate_io_pairs(
    module_name: str,
    functions: dict[str, P.Function],
    batch_no: int,
    num_reps: int = 30,
) -> tuple[ResultType, list[str]]:
    module = importlib.import_module(module_name)

    importlib.reload(module)

    seed = (batch_no << 10) ^ MAGIC_NUMBER
    datagen = DG.make_generator(seed)

    results: ResultType = {}
    bad_functions: list[str] = []

    for it, (function_hash, function_item) in enumerate(functions.items()):
        print(f"{it}: {function_hash}", end=" - ")
        inputs = []
        outputs = []
        for _ in range(num_reps):
            func_exe = getattr(module, f"func_{function_hash}")
            ctx = DG.DataGenContext()
            current_inputs = tuple(
                datagen.generate(ctx, arg.dtype) for arg in function_item.args
            )
            try:
                current_output = func_exe(*current_inputs)
            except Exception as e:
                print("[ERROR]", function_hash, e.__class__.__name__, e)
                bad_functions.append(function_hash)
                break
            else:
                if callable(current_output):
                    print(f"[ERROR] got callable output: {current_output}")
                    bad_functions.append(function_hash)
                    break
                inputs.append(current_inputs)
                outputs.append(current_output)
        else:
            if inputs and outputs:
                results[function_hash] = {"inputs": inputs, "outputs": outputs}
                print("GOOD")
            else:
                bad_functions.append(function_hash)
                print("BAD")

    return results, bad_functions


def filter_and_rewrite(
    filename: str, functions: dict[str, P.Function], bad_list: list[str]
) -> dict[str, P.Function]:
    good_functions = {
        key: value for key, value in functions.items() if key not in bad_list
    }
    write_functions(filename, good_functions.values())
    print(f"GOOD FUNCTIONS #: {len(good_functions)}")
    return good_functions


def write_to_json(filename: str, results: dict):

    def set_default(obj):
        if isinstance(obj, frozenset):
            return list(obj)
        raise TypeError(str(obj))

    with open(filename, "w") as json_file:
        json.dump(results, json_file, indent=None, default=set_default)


def report_metric(functions: dict[str, P.Function], results: ResultType) -> float:
    metric = Metric()
    scores = [
        metric.score(functions[fid], results[fid]["inputs"], results[fid]["outputs"])
        for fid in functions
    ]
    mean = np.mean(scores).item()
    print(f"Mean score = {mean:.4f}")
    return mean


def evaluate(
    sampler: Gen.Sampler,
    num_functions: int,
    grammar: Gen.Grammar,
    batch_no: int,
    io_reps: int,
    prefix: str = "",
    postifx: str = "",
    *,
    write_json: bool = True,
) -> float:
    use_nn = isinstance(sampler, NN.NeuralSampler)
    py_file_name = f"{prefix}generated{'_NN'if use_nn else ''}_{batch_no}{postifx}"
    gen_py_file = f"{GEN_DIR_NAME}/{py_file_name}.py"
    gen_io_file = (
        f"{GEN_DIR_NAME}/{prefix}io{'_NN' if use_nn else ''}_{batch_no}{postifx}.json"
    )

    functions = generate_functions(sampler, num_functions, grammar, batch_no)
    write_functions(gen_py_file, functions.values())

    module_name = f"{GEN_DIR_NAME.replace("/", ".")}.{py_file_name}"
    results, bad_list = generate_io_pairs(module_name, functions, batch_no, io_reps)

    functions = filter_and_rewrite(gen_py_file, functions, bad_list)
    score = report_metric(functions, results)

    if write_json:
        write_to_json(gen_io_file, results)

    return score


def train(
    sampler: NN.NeuralSampler,
    steps: int,
    grammar: Gen.Grammar,
    batch_no: int,
    prefix: str,
):

    seed = (batch_no << 10) ^ MAGIC_NUMBER
    datagen = DG.make_generator(seed)

    opt = optim.Adam(sampler.model.parameters(), lr=1e-4)
    metric = Metric()

    best_ma = -1
    sampler.model.train()
    for step in range(1, steps + 1):
        seed = ((batch_no << 10) | (step << 5)) ^ MAGIC_NUMBER
        sampler.reseed(seed)

        scores = []
        histories = []
        while len(scores) < 3:
            try:
                ctx = Gen.Context(grammar=grammar)
                gen = Gen.FunctionGenerator(sampler)
                function = gen.generate(ctx, "func")
                if function and function.args:
                    func_exe = compile_function(function)
                    inputs, outputs = generate_pairs(datagen, func_exe, function, 30)
                    scores.append(metric.score(function, inputs, outputs))
                    histories.append(torch.cat(sampler.model.scores))
            except Gen.TooMuchNesting:
                pass
            except Exception as e:
                print(f"  [ERROR]: {e.__class__.__name__}, {e!s}")
            else:
                pass

            sampler.reset_state()

        best_idx = np.argmax(scores).item()
        worst_idx = np.argmin(scores).item()
        num = histories[worst_idx].mean()
        den = histories[best_idx].mean()
        loss = num / den

        opt.zero_grad()
        loss.backward()
        opt.step()

        best_ma = scores[best_idx] if best_ma < 0 else (best_ma + scores[best_idx]) / 2
        print(f"step: {step}, loss = {loss.item():.4g}, best_score: {best_ma:.4g},")

    torch.save(sampler.model, f"{MODEL_DIR_NAME}/m_{prefix}_{batch_no}.pt")
    sampler.model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Program generator")
    parser.add_argument("batch_start", type=int)
    parser.add_argument("-c", "--batch-count", type=int, default=1)
    parser.add_argument("-n", "--neural", action="store_true", default=False)
    parser.add_argument("-t", "--train", type=int, default=0)
    parser.add_argument("-p", "--path", type=str, default="")
    parser.add_argument("-f", "--num-functions", type=int, default=1000)
    parser.add_argument("-i", "--io-reps", type=int, default=30)
    parser.add_argument("--dh", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--de", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--dm", type=int, default=256, help="Matching dimension")

    args = parser.parse_args()
    print(
        args.batch_start,
        args.batch_count,
        args.neural,
        args.train,
        args.path,
        args.num_functions,
        args.io_reps,
        args.dh,
        args.de,
        args.dm,
        sep=" | ",
    )

    batch_start: int = args.batch_start
    batch_count: int = args.batch_count
    train_steps: int = args.train
    num_functions: int = args.num_functions
    io_reps: int = args.io_reps
    use_neural: bool = args.train or args.neural
    model_path: str = args.path

    hidden_dim = args.dh
    embedding_dim = args.de
    matching_dim = args.dm

    grammar = Gen.Grammar(G, MODULE_ALIAS)

    date_time_str = datetime.datetime.now().strftime("%m%d%H%M")

    if model_path != "":
        sampler = NN.NeuralSampler(
            hidden_dim=hidden_dim, embed_dim=embedding_dim, match_dim=matching_dim
        )
        sampler.model = torch.load(model_path, weights_only=False)
        sampler.model.eval()
        print(f"Loaded model from {model_path}")

        evaluate(
            sampler,
            num_functions,
            grammar,
            batch_no=batch_start,
            io_reps=io_reps,
            prefix=f"{date_time_str}_",
            write_json=False,
        )

        exit(0)

    logs: list[str] = []
    for batch in range(batch_start, batch_start + batch_count):
        print(f"Starting batch: {batch}".center(99, "-"))
        sampler = (
            NN.NeuralSampler(
                hidden_dim=hidden_dim, embed_dim=embedding_dim, match_dim=matching_dim
            )
            if use_neural
            else Gen.Sampler()
        )
        before = evaluate(
            sampler,
            num_functions,
            grammar,
            batch,
            io_reps,
            prefix=f"{date_time_str}_",
            write_json=True,
        )
        logs.append(f"batch: {batch}, score_before: {before}")

        if train_steps and isinstance(sampler, NN.NeuralSampler):
            print()
            print(f"[{batch}] Start training for {train_steps}".center(99, "-"))
            train(sampler, train_steps, grammar, batch, date_time_str)
            after = evaluate(
                sampler,
                num_functions,
                grammar,
                batch,
                io_reps,
                prefix=f"{date_time_str}_",
                postifx="_after",
                write_json=True,
            )

            print(f"------->> Score>> before: {before:.4g}, after: {after:.4g}")
            logs.append(f"batch: {batch}, score_after: {after}")

    with open(
        f"{LOG_DIR_NAME}/log_{date_time_str}_b{batch_start}_{batch_count}.log", "w"
    ) as log:
        for line in logs:
            log.write(line)
            log.write("\n")
