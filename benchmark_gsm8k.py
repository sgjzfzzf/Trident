#!/usr/bin/env python3
"""Benchmark Qwen3-8B generation latency on GSM8K.

Loads the local Qwen3-8B model, optionally enables FlagGems with
``slice.Tensor`` disabled, and measures per-example generation latency for:

* 10 warmup examples
* 10 normal examples

Each generation is run with greedy decoding and ``max_new_tokens`` is 128 by
default. Results are printed to stdout and saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch


MODEL_PATH = "/home/swj/Trident/models/Qwen3-8B"
GSM8K_TEST_PARQUET = (
    "/root/.cache/modelscope/hub/datasets/downloads/"
    "d3f11db2e9a4ebe2d3c54065da591fb05b15378b128e416a881d9b4c075ef18d"
)
FLAGGEMS_WHITELIST = [
    "lt_scalar",
    "lt",
    "masked_fill",
    "rsqrt",
    "rsub_scalar",
    "rsub_tensor",
    "silu",
    "zero_",
    "add",
    "pow_tensor_scalar",
    "pow_tensor_tensor",
    "softmax",
    "embedding",
    "fill_scalar_",
    "bmm",
    "linear",
    "neg",
    "floor_divide",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def percentile(values: list[float], p: float) -> float:
    """Return a linear-interpolated percentile, useful for p50/p99."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def load_gsm8k_test() -> list[dict[str, str]]:
    from datasets import load_dataset

    if not Path(GSM8K_TEST_PARQUET).exists():
        raise FileNotFoundError(f"GSM8K test parquet not found: {GSM8K_TEST_PARQUET}")

    ds = load_dataset("parquet", data_files=GSM8K_TEST_PARQUET, split="train")
    return [
        {"question": row["question"], "answer": row["answer"]}
        for row in ds
    ]


def build_inputs(tokenizer, question: str, device: torch.device) -> dict[str, Any]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text)!r}, expected str")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs


def generate_one(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    inputs = build_inputs(tokenizer, question, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get(
        "attention_mask",
        torch.ones_like(input_ids, device=device),
    )
    prompt_len = int(input_ids.shape[1])

    # Prefill: one forward over the whole prompt.
    torch.cuda.synchronize(device)
    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize(device)
    prefill_time = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated_ids = [next_token]
    step_times: list[float] = []

    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones_like(next_token, device=device),
        ],
        dim=-1,
    )

    # Decode: greedy generation for the remaining max_new_tokens-1 tokens.
    for _ in range(max_new_tokens - 1):
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated_ids.append(next_token)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(next_token, device=device),
            ],
            dim=-1,
        )

    generated_tensor = torch.cat(generated_ids, dim=1)
    generated_len = int(generated_tensor.shape[1])
    answer = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)

    decode_time = sum(step_times)
    total_time = prefill_time + decode_time
    first_token_latency = prefill_time + (step_times[0] if step_times else 0.0)

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_len,
        "prefill_time_s": round(prefill_time, 6),
        "decode_time_s": round(decode_time, 6),
        "first_token_latency_s": round(first_token_latency, 6),
        "latency_s": round(total_time, 6),
        "tpop_s": round(decode_time / generated_len, 6) if generated_len else 0.0,
        "decode_step_times": [round(value, 6) for value in step_times],
        "decode_avg_per_step_s": round(decode_time / len(step_times), 6)
        if step_times
        else 0.0,
        "decode_tokens_per_second": round(generated_len / decode_time, 3)
        if decode_time > 0
        else 0.0,
        "tokens_per_second": round(generated_len / total_time, 3)
        if total_time > 0
        else 0.0,
        "question": question,
        "answer_preview": answer[:200],
    }


def run_benchmark(
    model,
    tokenizer,
    examples: list[dict[str, str]],
    warmup_count: int,
    normal_count: int,
    max_new_tokens: int,
    log_interval: int = 64,
    incremental_log_file: str = "/home/swj/Trident/gsm8k_bench_incremental.jsonl",
) -> dict[str, Any]:
    device = next(model.parameters()).device
    warmup_examples = examples[:warmup_count]
    normal_examples = examples[warmup_count : warmup_count + normal_count]

    if len(warmup_examples) < warmup_count or len(normal_examples) < normal_count:
        raise ValueError("Not enough GSM8K examples for requested counts")

    log_interval = max(1, log_interval)
    incremental_path = Path(incremental_log_file)
    incremental_path.parent.mkdir(parents=True, exist_ok=True)

    batch_pending: list[dict[str, Any]] = []
    next_result_index = 1
    batch_start_index = 1
    batch_started = time.perf_counter()
    batch_written = 0

    def write_incremental_batch(final: bool = False) -> None:
        nonlocal batch_pending, batch_start_index, batch_started, batch_written
        if not batch_pending or (not final and len(batch_pending) < log_interval):
            return

        latencies = [r["latency_s"] for r in batch_pending]
        prefills = [r["prefill_time_s"] for r in batch_pending]
        decodes = [r["decode_time_s"] for r in batch_pending]
        first_tokens = [r["first_token_latency_s"] for r in batch_pending]
        generated_tokens = [r["generated_tokens"] for r in batch_pending]
        tpop_values = [r["tpop_s"] for r in batch_pending]
        itl_values = [
            step
            for result in batch_pending
            for step in result.get("decode_step_times", [])
        ]
        wall_time = time.perf_counter() - batch_started
        total_generated_tokens = sum(generated_tokens)
        batch_written += 1
        record = {
            "batch_id": batch_written,
            "start_index": batch_start_index,
            "end_index": batch_start_index + len(batch_pending) - 1,
            "count": len(batch_pending),
            "total_generated_tokens": total_generated_tokens,
            "wall_time_s": round(wall_time, 6),
            "throughput_tokens_per_second": round(
                total_generated_tokens / wall_time, 3
            )
            if wall_time > 0
            else 0.0,
            "sum_latency_s": round(sum(latencies), 6),
            "mean_latency_s": round(statistics.mean(latencies), 6),
            "latency_p50_s": percentile(latencies, 50),
            "latency_p99_s": percentile(latencies, 99),
            "mean_prefill_time_s": round(statistics.mean(prefills), 6),
            "mean_decode_time_s": round(statistics.mean(decodes), 6),
            "mean_first_token_latency_s": round(statistics.mean(first_tokens), 6),
            "ttft_p50_s": percentile(first_tokens, 50),
            "ttft_p99_s": percentile(first_tokens, 99),
            "mean_tpop_s": round(statistics.mean(tpop_values), 6),
            "tpop_p50_s": percentile(tpop_values, 50),
            "tpop_p99_s": percentile(tpop_values, 99),
            "itl_p50_s": percentile(itl_values, 50),
            "itl_p99_s": percentile(itl_values, 99),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with incremental_path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            f.flush()
        log(
            f"Incremental batch written: batch={record['batch_id']} "
            f"range={record['start_index']}-{record['end_index']} "
            f"count={record['count']} wall={record['wall_time_s']:.4f}s "
            f"throughput={record['throughput_tokens_per_second']:.2f} tok/s "
            f"latency_p50={record['latency_p50_s']:.4f}s "
            f"ttft_p50={record['ttft_p50_s']:.4f}s "
            f"tpop_p50={record['tpop_p50_s']:.4f}s "
            f"itl_p50={record['itl_p50_s']:.4f}s "
            f"itl_p99={record['itl_p99_s']:.4f}s"
        )
        batch_pending = []
        batch_started = time.perf_counter()

    log("Starting warmup runs ...")
    warmup_results = []
    for i, example in enumerate(warmup_examples, start=1):
        log(f"[warmup {i}/{warmup_count}] question: {example['question'][:80]!r}")
        result = generate_one(
            model,
            tokenizer,
            example["question"],
            max_new_tokens,
            device,
        )
        result["index"] = i
        result["split"] = "warmup"
        warmup_results.append(result)
        if not batch_pending:
            batch_start_index = next_result_index
            batch_started = time.perf_counter()
        batch_pending.append(result)
        next_result_index += 1
        write_incremental_batch()
        log(
            f"[warmup {i}/{warmup_count}] prompt={result['prompt_tokens']} "
            f"generated={result['generated_tokens']} "
            f"prefill={result['prefill_time_s']:.4f}s "
            f"decode={result['decode_time_s']:.4f}s "
            f"first_token={result['first_token_latency_s']:.4f}s "
            f"latency={result['latency_s']:.4f}s "
            f"decode_throughput={result['decode_tokens_per_second']:.2f} tok/s"
        )

    log("Starting normal runs ...")
    normal_results = []
    for i, example in enumerate(normal_examples, start=1):
        log(f"[normal {i}/{normal_count}] question: {example['question'][:80]!r}")
        result = generate_one(
            model,
            tokenizer,
            example["question"],
            max_new_tokens,
            device,
        )
        result["index"] = i
        result["split"] = "normal"
        normal_results.append(result)
        if not batch_pending:
            batch_start_index = next_result_index
            batch_started = time.perf_counter()
        batch_pending.append(result)
        next_result_index += 1
        write_incremental_batch()
        log(
            f"[normal {i}/{normal_count}] prompt={result['prompt_tokens']} "
            f"generated={result['generated_tokens']} "
            f"prefill={result['prefill_time_s']:.4f}s "
            f"decode={result['decode_time_s']:.4f}s "
            f"first_token={result['first_token_latency_s']:.4f}s "
            f"latency={result['latency_s']:.4f}s "
            f"decode_throughput={result['decode_tokens_per_second']:.2f} tok/s"
        )

    write_incremental_batch(final=True)

    def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            return {
                "mean_latency_s": 0.0,
                "median_latency_s": 0.0,
                "min_latency_s": 0.0,
                "max_latency_s": 0.0,
                "mean_prefill_time_s": 0.0,
                "median_prefill_time_s": 0.0,
                "mean_decode_time_s": 0.0,
                "median_decode_time_s": 0.0,
                "mean_first_token_latency_s": 0.0,
                "median_first_token_latency_s": 0.0,
                "total_generated_tokens": 0,
                "throughput_tokens_per_second": 0.0,
                "latency_p50_s": 0.0,
                "latency_p99_s": 0.0,
                "ttft_p50_s": 0.0,
                "ttft_p99_s": 0.0,
                "mean_tpop_s": 0.0,
                "tpop_p50_s": 0.0,
                "tpop_p99_s": 0.0,
                "itl_p50_s": 0.0,
                "itl_p99_s": 0.0,
            }
        latencies = [r["latency_s"] for r in results]
        prefills = [r["prefill_time_s"] for r in results]
        decodes = [r["decode_time_s"] for r in results]
        first_tokens = [r["first_token_latency_s"] for r in results]
        tpop_values = [r["tpop_s"] for r in results]
        itl_values = [
            step
            for result in results
            for step in result.get("decode_step_times", [])
        ]
        total_generated_tokens = sum(r["generated_tokens"] for r in results)
        total_wall_time = sum(r["latency_s"] for r in results)
        return {
            "mean_latency_s": round(statistics.mean(latencies), 6),
            "median_latency_s": round(statistics.median(latencies), 6),
            "min_latency_s": round(min(latencies), 6),
            "max_latency_s": round(max(latencies), 6),
            "mean_prefill_time_s": round(statistics.mean(prefills), 6),
            "median_prefill_time_s": round(statistics.median(prefills), 6),
            "mean_decode_time_s": round(statistics.mean(decodes), 6),
            "median_decode_time_s": round(statistics.median(decodes), 6),
            "mean_first_token_latency_s": round(statistics.mean(first_tokens), 6),
            "median_first_token_latency_s": round(statistics.median(first_tokens), 6),
            "total_generated_tokens": total_generated_tokens,
            "throughput_tokens_per_second": round(
                total_generated_tokens / total_wall_time, 3
            )
            if total_wall_time > 0
            else 0.0,
            "latency_p50_s": percentile(latencies, 50),
            "latency_p99_s": percentile(latencies, 99),
            "ttft_p50_s": percentile(first_tokens, 50),
            "ttft_p99_s": percentile(first_tokens, 99),
            "mean_tpop_s": round(statistics.mean(tpop_values), 6),
            "tpop_p50_s": percentile(tpop_values, 50),
            "tpop_p99_s": percentile(tpop_values, 99),
            "itl_p50_s": percentile(itl_values, 50),
            "itl_p99_s": percentile(itl_values, 99),
        }

    summary = {
        "warmup": summarize(warmup_results),
        "normal": summarize(normal_results),
    }
    log("Warmup summary: " + json.dumps(summary["warmup"], ensure_ascii=False))
    log("Normal summary: " + json.dumps(summary["normal"], ensure_ascii=False))

    return {
        "summary": summary,
        "warmup_results": warmup_results,
        "normal_results": normal_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-count", type=int, default=10)
    parser.add_argument("--normal-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-flaggems", action="store_true")
    parser.add_argument("--result-file", default="/home/swj/Trident/gsm8k_bench_results.json")
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument(
        "--incremental-log-file",
        default="/home/swj/Trident/gsm8k_bench_incremental.jsonl",
    )
    args = parser.parse_args()

    log("=" * 80)
    log("Qwen3-8B GSM8K latency benchmark")
    log(f"python: {__import__('sys').version.split()[0]}")
    log(f"torch: {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}")
        log(f"gpu memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    if not args.no_flaggems:
        try:
            import flag_gems

            log("Enabling FlagGems white-list ...")
            flag_gems.only_enable(include=FLAGGEMS_WHITELIST)
            log(
                f"FlagGems enabled from {getattr(flag_gems, '__file__', '?')} "
                f"with {len(FLAGGEMS_WHITELIST)} requested operators"
            )
        except Exception:
            log("flag_gems.only_enable() failed")
            traceback.print_exc()
            return 2
    else:
        log("FlagGems disabled; running native PyTorch path.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Loading GSM8K test data ...")
    examples = load_gsm8k_test()
    log(f"Loaded {len(examples)} GSM8K test examples")

    log(f"Loading tokenizer/model from {MODEL_PATH} ...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    log(f"Model loaded in {time.time() - load_start:.1f}s")

    try:
        result = run_benchmark(
            model,
            tokenizer,
            examples,
            args.warmup_count,
            args.normal_count,
            args.max_new_tokens,
            log_interval=args.log_interval,
            incremental_log_file=args.incremental_log_file,
        )
    except Exception:
        log("Benchmark failed with exception")
        traceback.print_exc()
        return 3

    result["config"] = {
        "model_path": MODEL_PATH,
        "dataset": "AI-ModelScope/gsm8k",
        "dataset_split": "test",
        "warmup_count": args.warmup_count,
        "normal_count": args.normal_count,
        "max_new_tokens": args.max_new_tokens,
        "log_interval": args.log_interval,
        "incremental_log_file": args.incremental_log_file,
        "flaggems_enabled": not args.no_flaggems,
        "torch_version": torch.__version__,
    }
    result_path = Path(args.result_file)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Results written to {result_path}")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
