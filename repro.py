#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request


DEFAULT_MODEL = os.environ.get(
    "VLLM_POC_G8_MODEL",
    "/home/neil/code/llm/Qwen2-0.5B-official",
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "VLLM_POC_G8_OUTPUT_ROOT",
    "/tmp/g8_prefix_reuse_semantic_official_runs",
)

CONTEXTS = [
    "Read it as either a fish name or a possessive phrase.",
    "The note came from a pronunciation exercise.",
    "The fish-market memo and the jewelry receipt used the same phrase.",
    "The sentence intentionally balanced a fish meaning against a jewelry meaning.",
]

PREFIX = "Planning note: the margin comment on the page ended with her. " * 12
QUESTION = (
    "Choices:\n"
    "A = the fish word herring\n"
    "B = the jewelry phrase her ring\n"
    "Respond with A or B only.\n"
    "Answer:"
)


@dataclass(frozen=True)
class RequestScenario:
    alias: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_tokens: int = 0
    ignore_eos: bool = False
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    structured_outputs: dict[str, Any] | None = None
    sampling_overrides: dict[str, Any] = field(default_factory=dict)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def repeated_cycle(tokens: list[str], repeat_count: int) -> str:
    return " ".join(tokens * max(1, int(repeat_count)))


def build_prompt_mode(
    mode: str,
    *,
    cycle_size: int = 4,
    repeat_count: int = 24,
    noise_every_n: int = 0,
) -> str:
    cycle_pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    spec_pool = ["spec", "decode", "repeat", "cycle", "draft", "verify"]

    if mode == "cycle":
        prompt = repeated_cycle(cycle_pool[: max(1, cycle_size)], repeat_count)
    elif mode == "reverse_cycle":
        prompt = repeated_cycle(
            list(reversed(cycle_pool[: max(1, cycle_size)])),
            repeat_count,
        )
    elif mode == "spec_repeat":
        prompt = repeated_cycle(spec_pool[:3], repeat_count)
    elif mode == "omega_repeat":
        prompt = repeated_cycle(["omega"], repeat_count)
    elif mode == "sharedprefix_repeat":
        prompt = repeated_cycle(["sharedprefix"], repeat_count)
    else:
        raise ValueError(f"Unsupported prompt mode: {mode}")

    if noise_every_n <= 0:
        return prompt
    tokens = prompt.split()
    noisy: list[str] = []
    for index, token in enumerate(tokens, start=1):
        noisy.append(token)
        if index % int(noise_every_n) == 0:
            noisy.append(f"noise_{index}")
    return " ".join(noisy)


def build_choice_prompt(
    choice_count: int,
    *,
    exemplar_repeat_count: int = 0,
    prompt_style: str = "answer_cue",
) -> tuple[str, dict[str, Any]]:
    labels = [
        "accept",
        "reject",
        "wait",
        "retry",
        "defer",
        "cache",
        "escalate",
        "drop",
    ][: max(1, int(choice_count))]
    if exemplar_repeat_count > 0:
        exemplar_labels = labels[: max(2, min(4, len(labels)))]
        if prompt_style == "cycle":
            prompt = repeated_cycle(exemplar_labels, exemplar_repeat_count)
        else:
            blocks = [f"answer: {label}" for label in exemplar_labels]
            repeated = " ".join(blocks * max(1, int(exemplar_repeat_count)))
            prompt = (
                "Follow the repeated pattern and output exactly one routing action. "
                f"{repeated} answer:"
            )
    else:
        prompt = "Pick exactly one routing action: " + ", ".join(labels) + "."
    return prompt, {"choice": labels}


def build_json_prompt(
    *,
    variant: str = "schema_flat",
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
    exemplar_repeat_count: int = 0,
) -> tuple[str, dict[str, Any]]:
    if variant == "schema_flat":
        prompt = "Return a compact JSON object with keys action and confidence."
        payload: dict[str, Any] = {"json_object": True}
    elif variant == "schema_nested_list":
        prompt = (
            "Return a compact JSON object with keys route, confidence, and flags. "
            "route must contain action and mode."
        )
        payload = {
            "json": {
                "type": "object",
                "required": ["route", "confidence", "flags"],
                "properties": {
                    "route": {
                        "type": "object",
                        "required": ["action", "mode"],
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["accept", "reject", "wait", "retry", "defer", "cache"],
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["fast", "safe", "balanced"],
                            },
                        },
                        "additionalProperties": False,
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["warm", "cold", "spec", "retry", "final"],
                        },
                        "minItems": 1,
                        "maxItems": 3,
                    },
                },
                "additionalProperties": False,
            }
        }
    else:
        raise ValueError(f"Unsupported json prompt variant: {variant}")

    if exemplar_repeat_count > 0:
        exemplars = (
            'json: {"route":{"action":"accept","mode":"fast"},"confidence":81,"flags":["warm"]} '
            'json: {"route":{"action":"reject","mode":"safe"},"confidence":12,"flags":["retry"]} '
            'json: {"route":{"action":"defer","mode":"balanced"},"confidence":44,"flags":["spec"]}'
        )
        prompt = f"{prompt} " + " ".join([exemplars] * max(1, int(exemplar_repeat_count)))

    if "json" in payload:
        payload["disable_any_whitespace"] = bool(disable_any_whitespace)
        payload["disable_additional_properties"] = bool(disable_additional_properties)
    return prompt, payload


def resolve_prompt_and_structured_outputs(payload: dict[str, Any]) -> tuple[str, Any]:
    prompt = str(payload.get("prompt", ""))
    auto_structured_outputs = None

    prompt_builder = payload.get("prompt_builder")
    if isinstance(prompt_builder, dict):
        prompt = build_prompt_mode(
            str(prompt_builder.get("mode", "cycle")),
            cycle_size=int(prompt_builder.get("cycle_size", 4)),
            repeat_count=int(prompt_builder.get("repeat_count", 24)),
            noise_every_n=int(prompt_builder.get("noise_every_n", 0)),
        )

    choice_builder = payload.get("choice_builder")
    if isinstance(choice_builder, dict):
        prompt, auto_structured_outputs = build_choice_prompt(
            int(choice_builder.get("choice_count", 3)),
            exemplar_repeat_count=int(choice_builder.get("exemplar_repeat_count", 0)),
            prompt_style=str(choice_builder.get("prompt_style", "answer_cue")),
        )

    json_builder = payload.get("json_builder")
    if isinstance(json_builder, dict):
        prompt, auto_structured_outputs = build_json_prompt(
            variant=str(json_builder.get("variant", "schema_flat")),
            disable_any_whitespace=bool(json_builder.get("disable_any_whitespace", False)),
            disable_additional_properties=bool(
                json_builder.get("disable_additional_properties", False)
            ),
            exemplar_repeat_count=int(json_builder.get("exemplar_repeat_count", 0)),
        )

    prefix_builder = payload.get("prompt_prefix_builder")
    if isinstance(prefix_builder, dict):
        prefix_text = build_prompt_mode(
            str(prefix_builder.get("mode", "cycle")),
            cycle_size=int(prefix_builder.get("cycle_size", 4)),
            repeat_count=int(prefix_builder.get("repeat_count", 24)),
            noise_every_n=int(prefix_builder.get("noise_every_n", 0)),
        )
        prompt = f"{prefix_text} {prompt}".strip()

    suffix_builder = payload.get("prompt_suffix_builder")
    if isinstance(suffix_builder, dict):
        suffix_text = build_prompt_mode(
            str(suffix_builder.get("mode", "cycle")),
            cycle_size=int(suffix_builder.get("cycle_size", 4)),
            repeat_count=int(suffix_builder.get("repeat_count", 24)),
            noise_every_n=int(suffix_builder.get("noise_every_n", 0)),
        )
        prompt = f"{prompt} {suffix_text}".strip()

    literal_prefix = str(
        payload.get("prompt_prefix", "") or payload.get("prompt_literal_prefix", "")
    )
    literal_suffix = str(
        payload.get("prompt_suffix", "") or payload.get("prompt_literal_suffix", "")
    )
    if literal_prefix:
        prompt = f"{literal_prefix}{prompt}"
    if literal_suffix:
        prompt = f"{prompt}{literal_suffix}"

    structured_outputs = payload.get("structured_outputs")
    if structured_outputs is None:
        structured_outputs = auto_structured_outputs
    return prompt, structured_outputs


def merged_request_payload(
    template_payload: dict[str, Any],
    request_item: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(template_payload)
    merged.update(dict(request_item.get("overrides") or {}))
    for key, value in request_item.items():
        if key in {
            "request_id",
            "runtime_request_id",
            "logical_request_id",
            "resumable",
            "streaming_update",
            "template",
            "group",
            "notes",
            "overrides",
        }:
            continue
        merged[key] = value
    sampling_overrides = dict(template_payload.get("sampling_overrides") or {})
    sampling_overrides.update(dict(merged.get("sampling_overrides") or {}))
    merged["sampling_overrides"] = sampling_overrides
    return merged


def normalize_stop_value(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [] if not text else [text]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def normalize_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def build_sampling_overrides(base: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(base.get("sampling_overrides") or {})
    for source_key, target_key, normalizer in (
        ("min_p", "min_p", float),
        ("stop", "stop", normalize_stop_value),
        ("stop_token_ids", "stop_token_ids", normalize_int_list),
        ("include_stop_str_in_output", "include_stop_str_in_output", bool),
        ("bad_words", "bad_words", normalize_str_list),
        ("allowed_token_ids", "allowed_token_ids", normalize_int_list),
        ("truncate_prompt_tokens", "truncate_prompt_tokens", int),
        ("skip_special_tokens", "skip_special_tokens", bool),
        ("spaces_between_special_tokens", "spaces_between_special_tokens", bool),
        ("prompt_logprobs", "prompt_logprobs", int),
        ("logprobs", "logprobs", int),
        ("skip_reading_prefix_cache", "skip_reading_prefix_cache", bool),
    ):
        if source_key not in base:
            continue
        value = base[source_key]
        overrides[target_key] = None if value is None else normalizer(value)
    return overrides


def build_request_scenario(
    *,
    request_id: str,
    template_name: str,
    template_payload: dict[str, Any],
    request_item: dict[str, Any],
) -> tuple[RequestScenario, dict[str, Any]]:
    merged = merged_request_payload(template_payload, request_item)
    prompt, structured_outputs = resolve_prompt_and_structured_outputs(merged)
    scenario = RequestScenario(
        alias=request_id,
        prompt=prompt,
        max_tokens=int(merged.get("max_tokens", 16)),
        temperature=float(merged.get("temperature", 0.0)),
        top_p=float(merged.get("top_p", 1.0)),
        top_k=int(merged.get("top_k", 0)),
        min_tokens=int(merged.get("min_tokens", 0)),
        ignore_eos=bool(merged.get("ignore_eos", False)),
        repetition_penalty=float(merged.get("repetition_penalty", 1.0)),
        frequency_penalty=float(merged.get("frequency_penalty", 0.0)),
        presence_penalty=float(merged.get("presence_penalty", 0.0)),
        structured_outputs=structured_outputs,
        sampling_overrides=build_sampling_overrides(merged),
    )
    return scenario, merged


def build_sampling_params(
    scenario: RequestScenario,
    tokenizer: AutoTokenizer,
) -> SamplingParams:
    kwargs: dict[str, Any] = {
        "max_tokens": int(scenario.max_tokens),
        "temperature": float(scenario.temperature),
        "top_p": float(scenario.top_p),
        "top_k": int(scenario.top_k),
        "min_tokens": int(scenario.min_tokens),
        "ignore_eos": bool(scenario.ignore_eos),
        "repetition_penalty": float(scenario.repetition_penalty),
        "frequency_penalty": float(scenario.frequency_penalty),
        "presence_penalty": float(scenario.presence_penalty),
    }
    if scenario.structured_outputs is not None:
        kwargs["structured_outputs"] = StructuredOutputsParams(**scenario.structured_outputs)
    kwargs.update(dict(scenario.sampling_overrides or {}))
    params = SamplingParams(**kwargs)
    params.update_from_generation_config({}, tokenizer.eos_token_id)
    if not hasattr(tokenizer, "max_token_id"):
        setattr(tokenizer, "max_token_id", len(tokenizer) - 1)
    params.update_from_tokenizer(tokenizer)
    return params


def build_request(
    *,
    campaign_id: str,
    scenario: RequestScenario,
    prompt_token_ids: list[int],
    runtime_request_id: str | None,
    tokenizer: AutoTokenizer,
    block_hasher: Any,
    resumable: bool,
) -> tuple[str, Request]:
    runtime_alias = (
        str(runtime_request_id).strip()
        if runtime_request_id is not None
        else str(scenario.alias).strip()
    )
    if not runtime_alias:
        raise ValueError("runtime_request_id must not be empty")
    req_id = f"{campaign_id}_{runtime_alias}"
    request = Request(
        req_id,
        prompt_token_ids,
        build_sampling_params(scenario, tokenizer),
        None,
        block_hasher=block_hasher,
        resumable=bool(resumable),
    )
    return req_id, request


def build_engine(
    *,
    model: str,
    engine_cfg: dict[str, Any],
) -> tuple[Any, EngineCore]:
    kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": model,
        "skip_tokenizer_init": bool(engine_cfg.get("skip_tokenizer_init", False)),
        "enforce_eager": bool(engine_cfg.get("enforce_eager", True)),
        "distributed_executor_backend": "uni",
        "max_model_len": int(engine_cfg.get("max_model_len", 256)),
        "max_num_batched_tokens": int(engine_cfg.get("max_num_batched_tokens", 64)),
        "max_num_seqs": int(engine_cfg.get("max_num_seqs", 4)),
        "tensor_parallel_size": int(engine_cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(engine_cfg.get("gpu_memory_utilization", 0.25)),
        "kv_cache_memory_bytes": engine_cfg.get("kv_cache_memory_bytes"),
        "block_size": 16,
        "use_tqdm_on_load": False,
        "disable_custom_all_reduce": True,
        "async_scheduling": bool(engine_cfg.get("async_scheduling", False)),
        "seed": int(engine_cfg.get("seed", 0)),
    }
    if engine_cfg.get("speculative_config") is not None:
        kwargs["speculative_config"] = dict(engine_cfg["speculative_config"])
    if engine_cfg.get("structured_outputs_config") is not None:
        kwargs["structured_outputs_config"] = dict(engine_cfg["structured_outputs_config"])
    if engine_cfg.get("engine_overrides"):
        kwargs.update(dict(engine_cfg["engine_overrides"]))
    engine_args = EngineArgs(**kwargs)
    config = engine_args.create_engine_config(headless=True)
    engine = EngineCore(config, Executor.get_class(config), log_stats=False)
    return config, engine


def shutdown_engine(engine: EngineCore) -> None:
    try:
        engine.shutdown()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


def engine_has_pending_work(engine: EngineCore) -> bool:
    return bool(engine.scheduler.has_requests() or getattr(engine, "batch_queue", None))


def ensure_async_batch_queue(engine: EngineCore) -> None:
    if not getattr(engine, "async_scheduling", False):
        return
    if getattr(engine, "batch_queue", None) is not None:
        return
    batch_queue_size = max(2, int(getattr(engine, "batch_queue_size", 0) or 0))
    engine.batch_queue_size = batch_queue_size
    engine.batch_queue = deque(maxlen=batch_queue_size)


def step_engine(engine: EngineCore) -> tuple[dict[int, Any] | None, bool]:
    if not getattr(engine, "async_scheduling", False):
        return engine.step()
    ensure_async_batch_queue(engine)
    while True:
        outputs, executed = engine.step_with_batch_queue()
        if outputs is not None:
            return outputs, executed
        if not engine_has_pending_work(engine):
            return {}, executed


def summarize_engine_core_outputs(outputs: dict[int, Any]) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for engine_index, engine_outputs in sorted(outputs.items(), key=lambda item: item[0]):
        records: list[dict[str, Any]] = []
        for output in engine_outputs.outputs:
            records.append(
                {
                    "request_id": str(output.request_id),
                    "new_token_ids": [int(token_id) for token_id in list(output.new_token_ids)],
                    "finish_reason": (
                        None if output.finish_reason is None else str(output.finish_reason)
                    ),
                    "finished": bool(output.finished),
                    "text": "".join(
                        str(getattr(item, "text", ""))
                        for item in engine_outputs.outputs
                        if getattr(item, "request_id", None) == output.request_id
                    ),
                }
            )
        summary[str(engine_index)] = records
    return summary


def run_stateful_campaign(
    *,
    model: str,
    campaign: dict[str, Any],
    run_root: Path,
) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    config, engine = build_engine(model=model, engine_cfg=dict(campaign["engine"]))
    write_json(run_root / "campaign.json", campaign)

    req_aliases: dict[str, str] = {}
    active_runtime_req_ids: dict[str, str] = {}
    request_states: dict[str, dict[str, Any]] = {}
    action_log: list[dict[str, Any]] = []
    current_step = 0

    def update_request_states(outputs: dict[int, Any]) -> None:
        for records in summarize_engine_core_outputs(outputs).values():
            for record in records:
                runtime_request_id = str(record["request_id"])
                alias = req_aliases.get(runtime_request_id, runtime_request_id)
                state = request_states.setdefault(
                    alias,
                    {
                        "runtime_request_id": runtime_request_id,
                        "all_token_ids": [],
                        "new_token_batches": [],
                        "finish_reason": None,
                        "finished": False,
                    },
                )
                new_token_ids = [int(token_id) for token_id in list(record["new_token_ids"])]
                state["new_token_batches"].append(new_token_ids)
                state["all_token_ids"].extend(new_token_ids)
                state["text"] = str(state.get("text", "")) + str(record.get("text", ""))
                if record["finish_reason"] is not None:
                    state["finish_reason"] = str(record["finish_reason"])
                if bool(record["finished"]):
                    state["finished"] = True

    def resolve_cancel_runtime_request_ids(request_ids: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        for request_id in request_ids:
            runtime_request_id = active_runtime_req_ids.get(request_id)
            if runtime_request_id is None:
                state = request_states.get(request_id) or {}
                runtime_request_id = str(state.get("runtime_request_id") or "")
            if not runtime_request_id or runtime_request_id in seen:
                continue
            seen.add(runtime_request_id)
            resolved.append(runtime_request_id)
        return resolved

    def note_finished_request_ids(outputs: dict[int, Any]) -> None:
        for records in summarize_engine_core_outputs(outputs).values():
            for record in records:
                if bool(record["finished"]):
                    runtime_request_id = str(record["request_id"])
                    alias = req_aliases.get(runtime_request_id, runtime_request_id)
                    active_runtime_req_ids.pop(alias, None)

    def execute_one_step(label: str) -> bool:
        nonlocal current_step
        if not engine_has_pending_work(engine):
            return False
        outputs, executed = step_engine(engine)
        if outputs is None:
            return False
        update_request_states(outputs)
        note_finished_request_ids(outputs)
        if not getattr(engine, "async_scheduling", False):
            engine.post_step(executed)
        current_step += 1
        action_log.append(
            {
                "type": "engine_step",
                "label": label,
                "step_index": current_step,
                "executed": bool(executed),
                "outputs": summarize_engine_core_outputs(outputs),
            }
        )
        return True

    def run_async_queue_once(label: str) -> dict[str, Any]:
        nonlocal current_step
        ensure_async_batch_queue(engine)
        before_queue_len = len(engine.batch_queue or [])
        outputs, executed = engine.step_with_batch_queue()
        after_queue_len = len(engine.batch_queue or [])
        if outputs is not None:
            update_request_states(outputs)
            note_finished_request_ids(outputs)
            engine.post_step(executed)
            current_step += 1
        return {
            "produced_outputs": outputs is not None,
            "executed": bool(executed),
            "batch_queue_len_before": before_queue_len,
            "batch_queue_len_after": after_queue_len,
        }

    def run_async_queue_until_deferred(label: str, max_calls: int) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_calls))):
            result = run_async_queue_once(label)
            attempts.append(result)
            if not bool(result.get("produced_outputs", True)):
                break
        return {"attempt_count": len(attempts), "attempts": attempts}

    def drain_async_queue(label: str) -> dict[str, Any]:
        ensure_async_batch_queue(engine)
        if not engine.batch_queue:
            return {"drained": False, "reason": "batch_queue_empty"}
        future, scheduler_output, exec_future = engine.batch_queue.pop()
        model_output = future.result()
        if model_output is None:
            exec_future.result()
            raise RuntimeError("unexpected async queue model error")
        engine._process_aborts_queue()
        outputs = engine.scheduler.update_from_output(scheduler_output, model_output)
        executed = scheduler_output.total_num_scheduled_tokens > 0
        update_request_states(outputs)
        note_finished_request_ids(outputs)
        engine.post_step(executed)
        return {
            "drained": True,
            "queued_total_num_scheduled_tokens": int(
                scheduler_output.total_num_scheduled_tokens
            ),
            "outputs": summarize_engine_core_outputs(outputs),
        }

    try:
        for action_index, action in enumerate(list(campaign.get("actions") or [])):
            action_type = str(action.get("type", "")).strip().lower()
            label = str(action.get("label") or f"{action_type}_{action_index:02d}")
            record: dict[str, Any] = {
                "index": int(action_index),
                "type": action_type,
                "label": label,
                "step_before": current_step,
            }

            if action_type == "submit":
                submitted_request_ids: list[str] = []
                for request_item in list(action.get("requests") or []):
                    request_id = str(request_item["request_id"])
                    template_name = str(request_item["template"])
                    template_payload = dict((campaign.get("request_templates") or {})[template_name])
                    request_scenario, merged = build_request_scenario(
                        request_id=request_id,
                        template_name=template_name,
                        template_payload=template_payload,
                        request_item=dict(request_item),
                    )
                    prompt_token_ids = tokenizer.encode(
                        request_scenario.prompt,
                        add_special_tokens=False,
                    )
                    runtime_request_id = str(
                        request_item.get("runtime_request_id") or request_id
                    ).strip()
                    runtime_req_id, request = build_request(
                        campaign_id=str(campaign["campaign_id"]),
                        scenario=request_scenario,
                        prompt_token_ids=prompt_token_ids,
                        runtime_request_id=runtime_request_id,
                        tokenizer=tokenizer,
                        block_hasher=engine.request_block_hasher,
                        resumable=bool(request_item.get("resumable", False)),
                    )
                    if request.sampling_params is not None:
                        request.sampling_params.verify(
                            config.model_config,
                            config.speculative_config,
                            config.structured_outputs_config,
                            getattr(engine.structured_output_manager, "tokenizer", None),
                        )
                    if request.use_structured_output:
                        engine.structured_output_manager.grammar_init(request)
                    req_aliases[runtime_req_id] = request_id
                    engine.add_request(request)
                    active_runtime_req_ids[request_id] = runtime_req_id
                    request_states.setdefault(
                        request_id,
                        {
                            "runtime_request_id": runtime_req_id,
                            "template": template_name,
                            "group": merged.get("group"),
                            "all_token_ids": [],
                            "new_token_batches": [],
                            "finish_reason": None,
                            "finished": False,
                            "text": "",
                        },
                    )
                    submitted_request_ids.append(request_id)
                record["submitted_request_ids"] = submitted_request_ids

            elif action_type == "cancel":
                request_ids = [str(item) for item in list(action.get("request_ids") or [])]
                runtime_request_ids = resolve_cancel_runtime_request_ids(request_ids)
                engine.abort_requests(runtime_request_ids)
                if bool(action.get("release_aliases", False)):
                    for request_id in request_ids:
                        active_runtime_req_ids.pop(request_id, None)
                record["cancel_runtime_request_ids"] = runtime_request_ids

            elif action_type == "queued_cancel":
                request_ids = [str(item) for item in list(action.get("request_ids") or [])]
                runtime_request_ids = resolve_cancel_runtime_request_ids(request_ids)
                if runtime_request_ids:
                    engine.aborts_queue.put_nowait(runtime_request_ids)
                if bool(action.get("release_aliases", False)):
                    for request_id in request_ids:
                        active_runtime_req_ids.pop(request_id, None)
                record["cancel_runtime_request_ids"] = runtime_request_ids

            elif action_type == "run":
                executed_steps = 0
                for _ in range(max(0, int(action.get("steps", 1)))):
                    if not execute_one_step(label):
                        break
                    executed_steps += 1
                record["executed_steps"] = executed_steps

            elif action_type == "run_async_queue_until_deferred":
                record.update(
                    run_async_queue_until_deferred(label, int(action.get("max_calls", 8)))
                )

            elif action_type == "drain_async_queue":
                record.update(drain_async_queue(label))

            elif action_type == "barrier":
                executed_steps = 0
                max_steps = max(1, int(action.get("max_steps", 64)))
                while engine_has_pending_work(engine) and executed_steps < max_steps:
                    if not execute_one_step(label):
                        break
                    executed_steps += 1
                record["executed_steps"] = executed_steps
                record["drained"] = not engine_has_pending_work(engine)

            elif action_type == "assert_idle":
                record["passed"] = not engine_has_pending_work(engine)

            else:
                raise ValueError(f"Unsupported action type: {action_type}")

            record["step_after"] = current_step
            action_log.append(record)

        write_json(run_root / "action_log.json", action_log)
        write_json(run_root / "request_states.json", request_states)
        return 0
    except Exception:
        (run_root / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_root / "action_log.json", action_log)
        write_json(run_root / "request_states.json", request_states)
        return 1
    finally:
        shutdown_engine(engine)


def build_campaign(
    *,
    model: str,
    skip_prefix_cache: bool,
    campaign_id: str,
    description: str,
) -> dict[str, Any]:
    prefix_probe_template: dict[str, Any] = {
        "prompt": (
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "Planning note: the margin comment on the page ended with her. "
            "The final margin comment on the page ended with her ring. "
            "Which exact wordform is intended here, herring or her ring? "
            "Answer with exactly one of: herring | her ring. Answer:"
        ),
        "max_tokens": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_tokens": 3,
        "ignore_eos": True,
    }
    if skip_prefix_cache:
        prefix_probe_template["sampling_overrides"] = {
            "skip_reading_prefix_cache": True,
        }

    return {
        "campaign_id": campaign_id,
        "description": description,
        "model": model,
        "engine": {
            "max_model_len": 320,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 4,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.12,
            "kv_cache_memory_bytes": 33554432,
            "step_limit": 192,
            "skip_tokenizer_init": False,
            "quantization": None,
            "enforce_eager": True,
            "async_scheduling": True,
            "cudagraph_capture_sizes": None,
            "max_cudagraph_capture_size": None,
            "compilation_config": None,
            "seed": 0,
            "structured_outputs_config": None,
            "speculative_config": None,
            "auto_scale_kv_cache_memory_bytes": False,
            "preserve_max_model_len": True,
            "env_overrides": {},
            "engine_overrides": {"enable_prefix_caching": True},
        },
        "request_templates": {
            "spec_anchor": {
                "prompt_builder": {
                    "mode": "cycle",
                    "cycle_size": 4,
                    "repeat_count": 24,
                },
                "max_tokens": 27,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_tokens": 2,
                "ignore_eos": True,
            },
            "mid_pressure": {
                "prompt_builder": {
                    "mode": "reverse_cycle",
                    "cycle_size": 4,
                    "repeat_count": 24,
                },
                "max_tokens": 27,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 16,
                "min_tokens": 2,
                "ignore_eos": True,
            },
            "late_chunk0": {
                "prompt": (
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "Planning note: the margin comment on the page ended with her. "
                    "The final margin comment on the page ended with her"
                ),
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_tokens": 1,
                "ignore_eos": True,
            },
            "late_chunk1": {
                "prompt": (
                    "ring. Name the object described by the final expression above. "
                    "Reply with one word only: fish, ring, or other. Answer:"
                ),
                "max_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_tokens": 2,
                "ignore_eos": True,
            },
            "prefix_probe": prefix_probe_template,
            "prefix_probe_followup": {
                "prompt": (
                    "\nWhat category is that? Use a meaning category, not a part "
                    "of speech. One word only"
                ),
                "max_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_tokens": 1,
                "ignore_eos": True,
            },
            "prefix_probe_followup2": {
                "prompt": (
                    "\nNow map that last answer to one coarse category. Use object "
                    "for items or jewelry. Use food for fish or edible things. "
                    "One word only:"
                ),
                "max_tokens": 3,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_tokens": 1,
                "ignore_eos": True,
            },
        },
        "actions": [
            {
                "type": "submit",
                "label": "submit_anchor",
                "requests": [
                    {
                        "request_id": "req0",
                        "template": "spec_anchor",
                        "group": "anchor",
                    }
                ],
            },
            {"type": "run", "label": "anchor_warmup", "steps": 1},
            {
                "type": "submit",
                "label": "submit_mid_pressure",
                "requests": [
                    {
                        "request_id": "req1",
                        "template": "mid_pressure",
                        "group": "pressure",
                    }
                ],
            },
            {"type": "run", "label": "mid_overlap_window", "steps": 2},
            {
                "type": "submit",
                "label": "submit_late_chunk0",
                "requests": [
                    {
                        "request_id": "req2",
                        "template": "late_chunk0",
                        "group": "late_session",
                        "resumable": True,
                    }
                ],
            },
            {"type": "run", "label": "late_chunk0_window", "steps": 2},
            {
                "type": "submit",
                "label": "submit_late_chunk1_update",
                "requests": [
                    {
                        "request_id": "req2",
                        "template": "late_chunk1",
                        "group": "late_session",
                        "resumable": True,
                        "streaming_update": True,
                    }
                ],
            },
            {"type": "run", "label": "handoff_prequeue_window", "steps": 3},
            {
                "type": "submit",
                "label": "submit_prefix_probe",
                "requests": [
                    {
                        "request_id": "req3",
                        "template": "prefix_probe",
                        "group": "prefix_probe",
                        "resumable": True,
                    }
                ],
            },
            {"type": "run", "label": "probe_followup_window", "steps": 1},
            {
                "type": "submit",
                "label": "submit_prefix_probe_followup",
                "requests": [
                    {
                        "request_id": "req3",
                        "template": "prefix_probe_followup",
                        "group": "prefix_probe",
                        "resumable": True,
                        "streaming_update": True,
                    }
                ],
            },
            {"type": "run", "label": "probe_followup2_window", "steps": 3},
            {
                "type": "submit",
                "label": "submit_prefix_probe_followup2",
                "requests": [
                    {
                        "request_id": "req3",
                        "template": "prefix_probe_followup2",
                        "group": "prefix_probe",
                        "resumable": True,
                        "streaming_update": True,
                    }
                ],
            },
            {"type": "barrier", "label": "final_drain", "max_steps": 96},
            {"type": "assert_idle", "label": "final_idle"},
        ],
        "oracles": [
            {
                "type": "request_progress",
                "name": "anchor_progress",
                "group": "anchor",
                "min_total_tokens": 1,
            },
            {
                "type": "request_progress",
                "name": "pressure_progress",
                "group": "pressure",
                "min_total_tokens": 1,
            },
            {
                "type": "request_progress",
                "name": "late_session_progress",
                "group": "late_session",
                "min_total_tokens": 1,
            },
            {
                "type": "request_progress",
                "name": "prefix_probe_progress",
                "group": "prefix_probe",
                "min_total_tokens": 1,
            },
            {"type": "engine_idle", "name": "engine_idle_after_campaign"},
        ],
    }


def load_request_state(run_root: Path, request_id: str) -> dict[str, Any]:
    states = json.loads((run_root / "request_states.json").read_text(encoding="utf-8"))
    return dict(states.get(request_id) or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone official-model local reproducer for the G8 prefix-cache "
            "semantic divergence issue."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--run-name",
        default="g8_prefix_reuse_semantic_official_local_ctx1s1",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--context-index", type=int, default=1)
    parser.add_argument("--prequeue-steps", type=int, default=1)
    parser.add_argument(
        "--bias-b",
        type=float,
        default=0.0,
        help="Optional logit bias added to token ' B' in both hit/ctl runs.",
    )
    return parser.parse_args()


def build_official_campaign(
    *,
    model: str,
    tokenizer: AutoTokenizer,
    skip_prefix_cache: bool,
    context: str,
    prequeue_steps: int,
    bias_b: float,
    campaign_id: str,
) -> dict[str, Any]:
    campaign = build_campaign(
        model=model,
        skip_prefix_cache=skip_prefix_cache,
        campaign_id=campaign_id,
        description=campaign_id,
    )

    token_b = tokenizer.encode(" B", add_special_tokens=False)[0]
    sampling_overrides = {}
    if bias_b != 0.0:
        sampling_overrides["logit_bias"] = {token_b: float(bias_b)}

    campaign["request_templates"]["late_chunk0"]["prompt"] = (
        PREFIX + "The final margin comment on the page ended with her"
    )
    campaign["request_templates"]["late_chunk0"]["max_tokens"] = 1

    late_chunk1 = campaign["request_templates"]["late_chunk1"]
    late_chunk1["prompt"] = " ring. " + context + "\n" + QUESTION
    late_chunk1["max_tokens"] = 1
    late_chunk1["min_tokens"] = 1
    if sampling_overrides:
        late_chunk1["sampling_overrides"] = dict(sampling_overrides)

    prefix_probe = campaign["request_templates"]["prefix_probe"]
    prefix_probe["prompt"] = (
        PREFIX
        + "The final margin comment on the page ended with her ring. "
        + context
        + "\n"
        + QUESTION
    )
    prefix_probe["max_tokens"] = 1
    prefix_probe["min_tokens"] = 1
    if sampling_overrides:
        prefix_probe["sampling_overrides"] = dict(sampling_overrides)

    campaign["actions"] = [
        {
            "type": "submit",
            "label": "submit_anchor",
            "requests": [
                {
                    "request_id": "req0",
                    "template": "spec_anchor",
                    "group": "anchor",
                }
            ],
        },
        {"type": "run", "label": "anchor_warmup", "steps": 1},
        {
            "type": "submit",
            "label": "submit_mid_pressure",
            "requests": [
                {
                    "request_id": "req1",
                    "template": "mid_pressure",
                    "group": "pressure",
                }
            ],
        },
        {"type": "run", "label": "mid_overlap_window", "steps": 2},
        {
            "type": "submit",
            "label": "submit_late_chunk0",
            "requests": [
                {
                    "request_id": "req2",
                    "template": "late_chunk0",
                    "group": "late_session",
                    "resumable": True,
                }
            ],
        },
        {"type": "run", "label": "late_chunk0_window", "steps": 2},
        {
            "type": "submit",
            "label": "submit_late_chunk1_update",
            "requests": [
                {
                    "request_id": "req2",
                    "template": "late_chunk1",
                    "group": "late_session",
                    "resumable": True,
                    "streaming_update": True,
                }
            ],
        },
        {
            "type": "run",
            "label": "handoff_prequeue_window",
            "steps": int(prequeue_steps),
        },
        {
            "type": "submit",
            "label": "submit_prefix_probe",
            "requests": [
                {
                    "request_id": "req3",
                    "template": "prefix_probe",
                    "group": "prefix_probe",
                    "resumable": True,
                }
            ],
        },
        {"type": "barrier", "label": "final_drain", "max_steps": 64},
        {"type": "assert_idle", "label": "final_idle"},
    ]

    campaign["oracles"] = [
        {
            "type": "request_progress",
            "name": "anchor_progress",
            "group": "anchor",
            "min_total_tokens": 1,
        },
        {
            "type": "request_progress",
            "name": "pressure_progress",
            "group": "pressure",
            "min_total_tokens": 1,
        },
        {
            "type": "request_progress",
            "name": "late_session_progress",
            "group": "late_session",
            "min_total_tokens": 1,
        },
        {
            "type": "request_progress",
            "name": "prefix_probe_progress",
            "group": "prefix_probe",
            "min_total_tokens": 1,
        },
        {"type": "engine_idle", "name": "engine_idle_after_campaign"},
    ]
    return campaign


def main() -> int:
    args = parse_args()
    if args.context_index < 0 or args.context_index >= len(CONTEXTS):
        raise SystemExit(f"context-index must be in [0, {len(CONTEXTS) - 1}]")

    context = CONTEXTS[args.context_index]
    run_root = Path(args.output_root).expanduser().resolve() / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)
    hit_root = run_root / "hit"
    ctl_root = run_root / "ctl"
    hit_root.mkdir()
    ctl_root.mkdir()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    hit_campaign = build_official_campaign(
        model=args.model,
        tokenizer=tokenizer,
        skip_prefix_cache=False,
        context=context,
        prequeue_steps=args.prequeue_steps,
        bias_b=args.bias_b,
        campaign_id=f"{args.run_name}_hit",
    )
    ctl_campaign = build_official_campaign(
        model=args.model,
        tokenizer=tokenizer,
        skip_prefix_cache=True,
        context=context,
        prequeue_steps=args.prequeue_steps,
        bias_b=args.bias_b,
        campaign_id=f"{args.run_name}_ctl",
    )

    write_json(
        run_root / "repro_config.json",
        {
            "model": args.model,
            "run_name": args.run_name,
            "context_index": int(args.context_index),
            "context": context,
            "prequeue_steps": int(args.prequeue_steps),
            "bias_b": float(args.bias_b),
        },
    )
    write_json(hit_root / "campaign.json", hit_campaign)
    write_json(ctl_root / "campaign.json", ctl_campaign)

    hit_rc = run_stateful_campaign(
        model=args.model,
        campaign=hit_campaign,
        run_root=hit_root,
    )
    ctl_rc = run_stateful_campaign(
        model=args.model,
        campaign=ctl_campaign,
        run_root=ctl_root,
    )

    hit_state = load_request_state(hit_root, "req3")
    ctl_state = load_request_state(ctl_root, "req3")
    hit_ids = [int(token_id) for token_id in list(hit_state.get("all_token_ids") or [])]
    ctl_ids = [int(token_id) for token_id in list(ctl_state.get("all_token_ids") or [])]
    hit_text = tokenizer.decode(hit_ids, skip_special_tokens=False)
    ctl_text = tokenizer.decode(ctl_ids, skip_special_tokens=False)

    summary = {
        "hit_returncode": int(hit_rc),
        "ctl_returncode": int(ctl_rc),
        "hit_text": hit_text,
        "ctl_text": ctl_text,
        "hit_label": hit_text.strip(),
        "ctl_label": ctl_text.strip(),
        "label_map": {
            "A": "herring",
            "B": "her ring",
        },
        "hit_meaning": (
            "herring"
            if hit_text.strip() == "A"
            else "her ring" if hit_text.strip() == "B" else ""
        ),
        "ctl_meaning": (
            "herring"
            if ctl_text.strip() == "A"
            else "her ring" if ctl_text.strip() == "B" else ""
        ),
        "different": hit_text != ctl_text,
    }
    write_json(run_root / "semantic_summary.json", summary)

    if hit_rc != 0 or ctl_rc != 0:
        return 1
    if not (hit_text.strip() == "A" and ctl_text.strip() == "B"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
