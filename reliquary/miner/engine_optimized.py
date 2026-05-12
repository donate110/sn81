"""Optimized Miner Engine with parallel strategies.

Optimizations:
1. Parallel GRAIL Proofs: Batch forward passes for all rollouts
2. Parallel Prompt Strategy: Process multiple prompts concurrently  
3. Async Pipeline: Overlap generation, proof computation, and submission
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
import random as _random

import torch

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.miner.engine import (
    pick_prompt_idx,
    _compute_merkle_root,
    maybe_pull_checkpoint,
    _hf_download,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


class OptimizedMiningEngine:
    """Optimized mining engine with parallel proof generation and prompt processing."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        max_parallel_prompts: int = 3,
    ) -> None:
        """Initialize optimized mining engine.
        
        Args:
            max_parallel_prompts: Number of prompts to process concurrently
        """
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override
        self.max_parallel_prompts = max_parallel_prompts

        # Lazy imports for heavy deps
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        
        # Lock for checkpoint reloading
        self._checkpoint_lock = asyncio.Lock()
        self._loaded_checkpoint_path = None

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,
        use_drand: bool = True,
    ) -> list:
        """Parallel mining loop with concurrent prompt processing."""
        import httpx
        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import WindowState
        from reliquary import chain

        # Resolve validator URL
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # Compute randomness
        randomness = await self._compute_randomness(subtensor, 0, use_drand)

        rng = _random.Random()
        results = []
        local_n = 0
        local_hash = ""

        async with httpx.AsyncClient(timeout=60) as client:
            # Active tasks for parallel prompt processing
            active_tasks = []
            
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except (SubmissionError, Exception) as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pull new checkpoint if needed
                try:
                    async with self._checkpoint_lock:
                        local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                            state=state, local_n=local_n, local_hash=local_hash,
                            local_model=self.hf_model,
                            download_fn=_hf_download,
                            load_fn=self._load_checkpoint,
                        )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                if state.state != WindowState.OPEN:
                    # Wait for running tasks to complete
                    if active_tasks:
                        done, active_tasks = await asyncio.wait(
                            active_tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            try:
                                result = await task
                                if result:
                                    results.append(result)
                            except Exception as e:
                                logger.error("task failed: %s", e)
                    else:
                        await asyncio.sleep(1)
                    continue

                # Spawn new tasks up to max_parallel_prompts
                cooldown_set = set(state.cooldown_prompts)
                
                while len(active_tasks) < self.max_parallel_prompts:
                    try:
                        prompt_idx = pick_prompt_idx(self.env, cooldown_set, rng=rng)
                    except RuntimeError:
                        logger.info("env fully in cooldown")
                        break
                    
                    # Add to cooldown to prevent duplicate selection
                    cooldown_set.add(prompt_idx)
                    
                    # Spawn async task for this prompt
                    task = asyncio.create_task(
                        self._process_prompt(
                            prompt_idx, state.window_n, local_hash,
                            randomness, url, client
                        )
                    )
                    active_tasks.add(task)
                
                # Wait for at least one task to complete
                if active_tasks:
                    done, active_tasks = await asyncio.wait(
                        active_tasks, timeout=0.5, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        try:
                            result = await task
                            if result:
                                results.append(result)
                        except Exception as e:
                            logger.error("task failed: %s", e)
                else:
                    await asyncio.sleep(1)

        return results

    async def _process_prompt(
        self,
        prompt_idx: int,
        window_n: int,
        checkpoint_hash: str,
        randomness: str,
        validator_url: str,
        client,
    ):
        """Process a single prompt: generate, build proofs, submit."""
        from reliquary.miner.submitter import submit_batch_v2, SubmissionError
        
        try:
            problem = self.env.get_problem(prompt_idx)
            
            # Step 1: Generate rollouts (GPU bottleneck)
            generations = await asyncio.to_thread(
                self._generate_m_rollouts, problem, randomness
            )
            
            if len(generations) < M_ROLLOUTS:
                logger.warning(
                    "generated %d/%d for prompt %d; skipping",
                    len(generations), M_ROLLOUTS, prompt_idx,
                )
                return None

            # Step 2: Build proofs in parallel (batched forward pass)
            rollout_submissions = await asyncio.to_thread(
                self._build_rollout_submissions_parallel,
                generations, problem, randomness
            )
            
            # Step 3: Compute merkle root
            merkle_root = _compute_merkle_root(rollout_submissions)

            # Step 4: Submit
            request = BatchSubmissionRequest(
                miner_hotkey=self.wallet.hotkey.ss58_address,
                prompt_idx=prompt_idx,
                window_start=window_n,
                merkle_root=merkle_root,
                rollouts=rollout_submissions,
                checkpoint_hash=checkpoint_hash,
            )
            
            resp = await submit_batch_v2(validator_url, request, client=client)
            logger.info(
                "submitted window=%d prompt=%d accepted=%s reason=%s",
                window_n, prompt_idx, resp.accepted,
                resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
            )
            return resp
            
        except SubmissionError as exc:
            logger.error("submit failed for prompt %d: %s", prompt_idx, exc)
            return None
        except Exception:
            logger.exception("failed to process prompt %d", prompt_idx)
            return None

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions (same as original)."""
        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        eos = self.tokenizer.eos_token_id
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            try:
                first_eos = gen.index(eos)
                gen = gen[: first_eos + 1]
            except ValueError:
                pass
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submissions_parallel(
        self, generations: list[dict], problem, randomness: str
    ) -> list[RolloutSubmission]:
        """Build all rollout submissions with batched GRAIL proof generation.
        
        Optimization: Single forward pass for all rollouts instead of M sequential passes.
        """
        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        # Prepare all tokens and compute rewards
        all_tokens_list = []
        prompt_lengths = []
        rewards = []
        
        for gen in generations:
            all_tokens = gen["tokens"]
            prompt_length = gen["prompt_length"]
            completion_tokens = all_tokens[prompt_length:]
            completion_text = self.tokenizer.decode(completion_tokens)
            reward = self.env.compute_reward(problem, completion_text)
            
            all_tokens_list.append(all_tokens)
            prompt_lengths.append(prompt_length)
            rewards.append(reward)

        # OPTIMIZATION: Batch forward pass for all rollouts
        # Pad sequences to same length for batching
        max_len = max(len(tokens) for tokens in all_tokens_list)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        padded_tokens = []
        for tokens in all_tokens_list:
            padded = tokens + [pad_token_id] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        # Single batched forward pass
        proof_input = torch.tensor(
            padded_tokens, device=f"cuda:{self.proof_gpu}"
        )
        
        with torch.no_grad():
            hidden_states_batch, logits_batch = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )
        
        # Build commitments and submissions
        r_vec = self._verifier.generate_r_vec(randomness)
        model_name = getattr(self.hf_model, "name_or_path", "unknown")
        
        rollout_submissions = []
        
        for i, (all_tokens, prompt_length, reward) in enumerate(
            zip(all_tokens_list, prompt_lengths, rewards)
        ):
            # Extract this rollout's hidden states (unpadded)
            seq_len = len(all_tokens)
            hidden_states = hidden_states_batch[i, :seq_len]
            
            # Build commitments
            commitments = self._verifier.create_commitments_batch(
                hidden_states, r_vec
            )
            
            # Compute token log-probs (fp32 for precision)
            log_probs = torch.log_softmax(logits_batch[i, :seq_len].float(), dim=-1)
            token_logprobs = []
            for j in range(prompt_length, len(all_tokens)):
                token_logprobs.append(log_probs[j - 1, all_tokens[j]].item())
            
            # Sign
            signature = sign_commit_binding(
                all_tokens, randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )
            
            commit = {
                "tokens": all_tokens,
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": {
                    "prompt_length": prompt_length,
                    "completion_length": len(all_tokens) - prompt_length,
                    "success": True,
                    "total_reward": 0.0,
                    "advantage": 0.0,
                    "token_logprobs": token_logprobs,
                },
            }
            
            rollout_submissions.append(
                RolloutSubmission(
                    tokens=all_tokens,
                    reward=reward,
                    commit=commit,
                )
            )
        
        return rollout_submissions

    def _load_checkpoint(self, local_path: str):
        """Reload both models from local_path (same as original)."""
        import torch
        from transformers import AutoModelForCausalLM
        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # Reload hf_model (for GRAIL proofs)
        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception("Failed to reload hf_model from %s", local_path)
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        torch.cuda.empty_cache()

        # Reload vllm_model
        try:
            new_gen = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception("Failed to reload vllm_model from %s", local_path)
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        torch.cuda.empty_cache()

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness (same as original)."""
        from reliquary import chain
        
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        return chain.compute_window_randomness(block_hash)
