from typing import List, Optional
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from benchmarks_ours.data_sets.data_set import Data_set
from benchmarks_ours.evals.utils import str2class, set_seed
import os
from dataclasses import dataclass, field
import logging
logging.basicConfig(level=logging.INFO)
from collections import defaultdict
import numpy as np
from tqdm.contrib import tenumerate

logger = logging.getLogger(__name__)


@dataclass
class EvalConfigs:
    dataset: str
    model: str
    approach: str
    tot_num_data: int = 200
    all_datasets: List[str] = field(
        default_factory=lambda: [
            '2wikimqa',
            'musique',
            'samsum',
            'multi_news',
            'hotpotqa',
            'needle',
        ]
    )  # 'mmlu', 'passage_retrieval_en',
    all_models: List[str] = field(
        default_factory=lambda: [
            'Qwen/Qwen3-4B',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'meta-llama/Llama-3.1-70B-Instruct',
        ]
    )  # Example public model IDs. Any HF ID or local model path is accepted.

    all_approaches: List[str] = field(
        default_factory=lambda: [
            'fr',
            'naive',
            'cacheblend-20',
            'cacheblend-15',
            'cacheblend-10',
            'cacheblend-5',
            'cacheblend-1',
            'epic-64',
            'epic-32',
            'epic-16',
            'epic-8',
            'epic-4',
            'epic-2',
            'epic-1',
            'CSO',
            'SSO',
        ]
    )  # Make sure the first approach is the fr one

    model_config: AutoConfig = field(init=False)

    seed: int = 42
    result_path: str = "results"

    @classmethod
    def get_configs_from_cli_args(cls) -> "EvalConfigs":
        """
        Parse the command line arguments and return the Configs object.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--approach", type=str, required=True)

        args = parser.parse_args()
        args.approach = cls._normalize_approach(args.approach)
        configs = cls(**vars(args))
        return configs

    @staticmethod
    def _normalize_approach(approach: str) -> str:
        normalized = approach.strip()
        lower = normalized.lower()
        if lower in ("cso", "sso"):
            return lower.upper()
        return lower

    def __post_init__(self):
        """
        Verify the init arguments and create the result path.
        """
        tot_num_data_env = os.getenv("TOT_NUM_DATA")
        if tot_num_data_env is not None:
            self.tot_num_data = int(tot_num_data_env)
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset,
                                        self.model.split("/")[-1])
        os.makedirs(self.result_path, exist_ok=True)

        self.model_config = AutoConfig.from_pretrained(self.model)

    def _verify_init_args(self):
        assert self.dataset in self.all_datasets, f"{self.dataset} not in {self.all_datasets}"
        assert self.approach in self.all_approaches, f"{self.approach} not in {self.all_approaches}"


class EvalEngine:
    """
    Evaluate a specific approach on a specific model and a specific dataset.
    """

    def __init__(self, configs: EvalConfigs) -> None:
        self.configs = configs

    def run(self):
        logging.info(
            (
                f"Evaluate \033[32m{self.configs.approach}\033[0m on"
                f" \033[32m{self.configs.model}\033[0m and"
                f" \033[32m{self.configs.dataset}\033[0m"
            )
        )
        logging.info(f"Save the results to \033[32m{self.configs.result_path}\033[0m")

        set_seed(self.configs.seed)

        # Step 1: Preprocessing, load and modify necessary components such as
        # tokenizer, dataset, model and pipeline.

        # Some model tokenizers do not add a BOS token by default, so we add it here.
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.configs.model,
            add_bos_token=True,
        )
        self.dataset: Data_set = str2class[self.configs.dataset](
            tokenizer=self.tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        self.dataset.save_dataset(self.configs.result_path)  # checkpoint dataset
        gpu_memory_utilization = float(
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))
        max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
        enforce_eager_env = os.getenv("VLLM_ENFORCE_EAGER")
        tp_size_env = os.getenv("VLLM_TENSOR_PARALLEL_SIZE")
        llm_kwargs = {}
        if tp_size_env:
            tp_size = int(tp_size_env)
            if tp_size > 1 and self.configs.approach != "fr":
                raise ValueError(
                    "This eval pipeline currently assumes a single vLLM worker "
                    "(tensor_parallel_size=1) because it directly reads/writes "
                    "per-worker EPIC state (old_kvs/cache_fuse_metadata). "
                    "Please run with VLLM_TENSOR_PARALLEL_SIZE=1 for EPIC "
                    "approaches, or use approach=fr."
                )
            llm_kwargs["tensor_parallel_size"] = tp_size
        if max_model_len_env:
            llm_kwargs["max_model_len"] = int(max_model_len_env)
        if enforce_eager_env is not None:
            llm_kwargs["enforce_eager"] = enforce_eager_env.strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
        self.model: AutoModelForCausalLM = LLM(
            model=self.configs.model,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            **llm_kwargs,
        )
        self.model.set_tokenizer(self.tokenizer)

        # Step 2: Run the inference and record results into the dataset
        self.dataset = self.run_inference(self.configs, self.dataset,
                                          self.tokenizer, self.model)

        # Step 3: Generate the presentation
        self.generate_presentation()

    def run_inference(self, configs: EvalConfigs, dataset: Data_set,
                      tokenizer: AutoTokenizer, llm: LLM) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """

        logger.info("Run the inference. This might take a long time... Good luck")

        results = defaultdict(list)

        for sample_idx, (system_prompts, mod_prompts,
                         free_form_prompt) in tenumerate(dataset,
                                                         desc="dataset",
                                                         leave=True):
            logger.info(f"Processing sample {sample_idx}")

            # Clear metadata at start of each sample to prevent state leakage.
            cache_fuse_metadata = (
                llm.llm_engine.model_executor.driver_worker.model_runner.model.model
                .cache_fuse_metadata)
            cache_fuse_metadata["full_hidden_states"] = None
            cache_fuse_metadata["full_positions"] = None
            cache_fuse_metadata["imp_indices"] = None
            cache_fuse_metadata["fake_q"] = None
            cache_fuse_metadata["attn_bias"] = None

            # CSO/SSO metadata (reset every sample)
            cache_fuse_metadata["suffix_len"] = 1
            cache_fuse_metadata["use_anchor_tokens"] = False
            cache_fuse_metadata["overlap_length"] = 0
            cache_fuse_metadata["anchor_boost_weight"] = 1.0
            cache_fuse_metadata["use_phase_alignment"] = False
            cache_fuse_metadata["alignment_period"] = 512
            cache_fuse_metadata["unified_rope_scaling"] = False
            cache_fuse_metadata["chunk_original_positions"] = None
            cache_fuse_metadata["target_positions"] = None
            cache_fuse_metadata["block_boundaries"] = None
            cache_fuse_metadata["num_blocks"] = None

            cache_fuse_metadata["epic"] = None
            cache_fuse_metadata["epic_used"] = False
            cache_fuse_metadata["check"] = False
            cache_fuse_metadata["collect"] = False
            cache_fuse_metadata["_epic_rope_pos_clamped"] = False

            # Convert modules in the prompt to tokens.
            system_token_ids: List[int] = tokenizer.encode(system_prompts)
            mod_token_ids: List[List[int]] = [
                tokenizer.encode(mod_prompt) for mod_prompt in mod_prompts
            ]
            free_form_token_ids: List[int] = tokenizer.encode(free_form_prompt)
            token_ids: List[List[int]] = [system_token_ids
                                          ] + mod_token_ids + [free_form_token_ids]

            # Include all BOS tokens.
            input_ids = [
                _token_ids[i] for _token_ids in token_ids
                for i in range(len(_token_ids))
            ]

            logger.debug("Collecting old kvs")
            cache_fuse_metadata["collect"] = True
            sampling_params = SamplingParams(temperature=0, max_tokens=1)

            is_cso_sso = configs.approach in ('CSO', 'SSO')
            is_sso = configs.approach == 'SSO'

            if is_cso_sso:
                # CSO/SSO defaults (aligned with epic-main scripts).
                cache_fuse_metadata["use_anchor_tokens"] = True
                cache_fuse_metadata["overlap_length"] = 16
                cache_fuse_metadata["anchor_boost_weight"] = 1.0
                cache_fuse_metadata["use_phase_alignment"] = True
                cache_fuse_metadata["alignment_period"] = 512
                cache_fuse_metadata["unified_rope_scaling"] = True
                cache_fuse_metadata["recomp_ratio"] = 0.0

            use_anchor_tokens = bool(cache_fuse_metadata.get("use_anchor_tokens",
                                                            False))
            overlap_length = int(cache_fuse_metadata.get("overlap_length", 0))
            use_phase_alignment = bool(
                cache_fuse_metadata.get("use_phase_alignment", False))
            alignment_period = int(
                cache_fuse_metadata.get("alignment_period", 512))

            old_kvs = []
            prev_chunk_token_ids: Optional[List[int]] = None
            chunk_original_positions: List[int] = []
            target_positions: List[int] = []
            block_boundaries: List[int] = []
            current_position = 0

            num_chunks_to_collect = len(token_ids) - (1 if is_cso_sso else 0)
            for i in range(num_chunks_to_collect):
                chunk_token_ids = token_ids[i]
                anchor_tokens: List[int] = []

                if (is_cso_sso and use_anchor_tokens and overlap_length > 0
                        and prev_chunk_token_ids is not None):
                    if is_sso:
                        head_pool = chunk_token_ids
                        if (head_pool and tokenizer.bos_token_id is not None
                                and head_pool[0] == tokenizer.bos_token_id):
                            head_pool = head_pool[1:]
                        anchor_tokens = head_pool[:overlap_length]
                    else:
                        anchor_tokens = prev_chunk_token_ids[-overlap_length:]

                anchor_offset = len(anchor_tokens)
                chunk_token_ids_with_anchor = anchor_tokens + chunk_token_ids

                # Reset per-request KV collection buffers to avoid mixing chunks
                # when chunked prefill is enabled.
                llm_layers = (
                    llm.llm_engine.model_executor.driver_worker.model_runner.model.model
                    .layers)
                for _layer in llm_layers:
                    _layer.self_attn.hack_kv = []

                llm.generate(
                    prompts={
                        "prompt_token_ids":
                        chunk_token_ids_with_anchor if is_cso_sso else chunk_token_ids
                    },
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                for j in range(len(llm_layers)):
                    if i == 0:
                        temp_k = llm_layers[j].self_attn.hack_kv[0].clone()
                        temp_v = llm_layers[j].self_attn.hack_kv[1].clone()
                    else:
                        if is_cso_sso and anchor_offset > 0:
                            temp_k = llm_layers[j].self_attn.hack_kv[0][
                                anchor_offset:anchor_offset + len(chunk_token_ids)
                            ].clone()
                            temp_v = llm_layers[j].self_attn.hack_kv[1][
                                anchor_offset:anchor_offset + len(chunk_token_ids)
                            ].clone()
                        else:
                            temp_k = llm_layers[j].self_attn.hack_kv[0].clone()
                            temp_v = llm_layers[j].self_attn.hack_kv[1].clone()

                    if i == 0:
                        old_kvs.append([temp_k, temp_v])
                    else:
                        old_kvs[j][0] = torch.cat((old_kvs[j][0], temp_k), dim=0)
                        old_kvs[j][1] = torch.cat((old_kvs[j][1], temp_v), dim=0)

                if is_cso_sso:
                    block_len = len(chunk_token_ids)
                    original_pos_start = anchor_offset
                    block_start = current_position
                    if use_phase_alignment:
                        aligned_start = (block_start //
                                         alignment_period) * alignment_period
                        target_base = aligned_start
                    else:
                        target_base = block_start
                    for pos_in_block in range(block_len):
                        chunk_original_positions.append(original_pos_start +
                                                        pos_in_block)
                        target_positions.append(target_base + pos_in_block)
                    current_position += block_len
                    if i < len(token_ids) - 1:
                        block_boundaries.append(current_position)
                    prev_chunk_token_ids = chunk_token_ids

            if is_cso_sso:
                cache_fuse_metadata["suffix_len"] = len(token_ids[-1])
                cache_fuse_metadata["block_boundaries"] = block_boundaries
                cache_fuse_metadata["num_blocks"] = len(token_ids)
                pos_device = (old_kvs[0][0].device if old_kvs and old_kvs[0]
                              and old_kvs[0][0] is not None else None)
                cache_fuse_metadata["chunk_original_positions"] = torch.tensor(
                    chunk_original_positions,
                    dtype=torch.long,
                    device=pos_device,
                )
                cache_fuse_metadata["target_positions"] = torch.tensor(
                    target_positions,
                    dtype=torch.long,
                    device=pos_device,
                )

                query_len = len(token_ids[-1])
                if query_len > 0:
                    for j in range(len(old_kvs)):
                        k_prefix = old_kvs[j][0]
                        v_prefix = old_kvs[j][1]
                        k_pad = torch.zeros(
                            (query_len, k_prefix.shape[1]),
                            dtype=k_prefix.dtype,
                            device=k_prefix.device,
                        )
                        v_pad = torch.zeros(
                            (query_len, v_prefix.shape[1]),
                            dtype=v_prefix.dtype,
                            device=v_prefix.device,
                        )
                        old_kvs[j][0] = torch.cat((k_prefix, k_pad), dim=0)
                        old_kvs[j][1] = torch.cat((v_prefix, v_pad), dim=0)

            expected_seq_len = sum(len(_token_ids) for _token_ids in token_ids)
            per_layer_lens = [
                (kv[0].shape[0] if kv and kv[0] is not None else -1)
                for kv in old_kvs
            ]
            actual_seq_len = per_layer_lens[0] if per_layer_lens else -1
            old_kvs_ok = per_layer_lens and all(
                l == expected_seq_len for l in per_layer_lens)
            if not old_kvs_ok:
                logger.warning(
                    "[EPIC DEBUG] Collected old_kvs length mismatch: "
                    f"expected={expected_seq_len}, actual={actual_seq_len}. "
                    "Disabling epic/check for this sample to avoid OOB.")

            llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = old_kvs

            logger.debug("Second inference for performance evaluation")
            start_offset = [0]
            for _token_ids in token_ids:
                start_offset.append(start_offset[-1] + len(_token_ids))

            if configs.approach == 'naive':
                if old_kvs_ok:
                    cache_fuse_metadata["epic"] = [start_offset[-1] - 1]
                else:
                    cache_fuse_metadata["epic"] = None
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["epic_used"] = False

            elif configs.approach.startswith('epic-'):
                recomp_num = int(configs.approach.split('-')[-1])
                temp: List[int] = []
                for i in range(1, len(start_offset) - 1):
                    if i == len(start_offset) - 2:
                        # Last document before query: recompute all tokens.
                        temp += list(range(start_offset[i], start_offset[i + 1]))
                    else:
                        end_idx = min(start_offset[i] + recomp_num,
                                      start_offset[i + 1])
                        temp += list(range(start_offset[i], end_idx))

                if old_kvs_ok:
                    cache_fuse_metadata["epic"] = temp
                else:
                    cache_fuse_metadata["epic"] = None
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["epic_used"] = False

            elif 'cacheblend' in configs.approach:
                recomp_ratio = int(configs.approach.split('-')[-1]) / 100
                cache_fuse_metadata["epic"] = None
                cache_fuse_metadata["check"] = bool(old_kvs_ok)
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["recomp_ratio"] = recomp_ratio

            elif configs.approach == 'fr':
                cache_fuse_metadata["epic"] = None
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata["collect"] = False

            elif configs.approach in ('CSO', 'SSO'):
                prefix_len = start_offset[-2]
                total_len = start_offset[-1]
                if old_kvs_ok:
                    cache_fuse_metadata["epic"] = list(range(prefix_len, total_len))
                else:
                    cache_fuse_metadata["epic"] = None
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["epic_used"] = False

            else:
                raise ValueError(f"Invalid approach: {configs.approach}")

            sampling_params = SamplingParams(
                temperature=0,
            )
            output = llm.generate(
                prompts={"prompt_token_ids": input_ids},
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            results[f'output_{configs.approach}'].append(output[0].outputs[0].text)
            results[f'TTFT_{configs.approach}'].append(
                output[0].metrics.first_token_time -
                output[0].metrics.first_scheduled_time)

        dataset.update(results)
        dataset.save_dataset(configs.result_path)
        return dataset

    def generate_presentation(self):
        """
        Present the results by invoking this function after executing run().
        The results are saved in self.configs.result_path.
        """
        self.dataset.calc_accuracy(self.configs.dataset, self.configs.approach)
        self.dataset.save_dataset(self.configs.result_path)

        accuracy_avg = np.mean(self.dataset.data[f"score_{self.configs.approach}"])
        ttft_avg = np.mean(self.dataset.data[f"TTFT_{self.configs.approach}"])
        logger.info(f"Average accuracy of {self.configs.approach}: {accuracy_avg:.6f}")
        logger.info(f"Average TTFT of {self.configs.approach}: {ttft_avg:.5f} s")


if __name__ == "__main__":
    configs = EvalConfigs.get_configs_from_cli_args()
    eval_engine = EvalEngine(configs)
    eval_engine.run()
