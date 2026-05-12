"""vLLM-accelerated mining engine with parallel optimizations.

This engine uses the vLLM library for high-performance generation:
- PagedAttention for efficient KV cache management
- Continuous batching for better GPU utilization  
- Optimized CUDA kernels for 2-10× faster generation

Requires: pip install vllm
Falls back to HuggingFace transformers if vLLM not available.
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

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning(
        "vLLM not available. Install with: pip install vllm\n"
        "Falling back to HuggingFace transformers (slower generation)."
    )


class VLLMMiningEngine:
    """Mining engine with vLLM for generation and HF for GRAIL proofs.
    
    Architecture:
    - vLLM LLM: Ultra-fast generation on GPU 0
    - HuggingFace model: GRAIL proof computation on GPU 1
    - Parallel prompts: Process multiple prompts concurrently
    - Batched proofs: Single forward pass for all rollouts
    """

    def __init__(
        self,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        model_path: str = None,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        max_parallel_prompts: int = 3,
        vllm_tensor_parallel_size: int = 1,
    ) -> None:
        """Initialize vLLM mining engine.
        
        Args:
            hf_model: HuggingFace model for GRAIL proofs
            tokenizer: Tokenizer
            wallet: Bittensor wallet
            env: Environment
            model_path: Path to model for vLLM (required)
            vllm_gpu: GPU for vLLM generation
            proof_gpu: GPU for GRAIL proofs
            max_new_tokens: Max tokens to generate
            validator_url_override: Override validator URL
            max_parallel_prompts: Concurrent prompts to process
            vllm_tensor_parallel_size: Number of GPUs for vLLM tensor parallelism
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM not available. Install with: pip install vllm\n"
                "Or use OptimizedMiningEngine for HuggingFace-based mining."
            )
        
        if model_path is None:
            raise ValueError("model_path required for vLLM engine")
        
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.model_path = model_path
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override
        self.max_parallel_prompts = max_parallel_prompts
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size

        # Initialize vLLM for generation
        logger.info("Initializing vLLM for fast generation on GPU %d...", vllm_gpu)
        self.vllm_engine = LLM(
            model=model_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        logger.info("vLLM initialized successfully")
        
        # Sampling params for generation (shared across all requests)
        self.sampling_params = SamplingParams(
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            max_tokens=max_new_tokens,
            n=M_ROLLOUTS,  # Generate M rollouts per prompt
        )

        # Lazy imports for heavy deps
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        
        # Lock for checkpoint reloading
        self._checkpoint_lock = asyncio.Lock()
        self._loaded_checkpoint_path = model_path

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,
        use_drand: bool = True,
    ) -> list:
        """Parallel mining loop with vLLM generation."""
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
        """Process a single prompt with vLLM generation."""
        from reliquary.miner.submitter import submit_batch_v2, SubmissionError
        
        try:
            problem = self.env.get_problem(prompt_idx)
            
            # Step 1: Generate rollouts with vLLM (FAST!)
            generations = await asyncio.to_thread(
                self._generate_m_rollouts_vllm, problem, randomness
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

    def _generate_m_rollouts_vllm(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions using vLLM.
        
        vLLM advantages:
        - PagedAttention: Efficient KV cache management
        - Continuous batching: Better GPU utilization
        - Optimized kernels: 2-10× faster than HuggingFace
        """
        prompt_text = problem["prompt"]
        
        # vLLM generates multiple outputs efficiently
        outputs = self.vllm_engine.generate(
            [prompt_text],
            self.sampling_params,
            use_tqdm=False,
        )
        
        # Extract rollouts from vLLM output
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        rollouts = []
        for output in outputs[0].outputs:
            # Get completion tokens
            completion_tokens = output.token_ids
            all_tokens = prompt_tokens + completion_tokens
            
            rollouts.append({
                "tokens": all_tokens,
                "prompt_length": prompt_length,
            })
        
        return rollouts

    def _build_rollout_submissions_parallel(
        self, generations: list[dict], problem, randomness: str
    ) -> list[RolloutSubmission]:
        """Build all rollout submissions with batched GRAIL proof generation."""
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
        """Reload HF model for proofs and reinitialize vLLM for generation."""
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

        # Reinitialize vLLM (no in-place reload, must recreate)
        try:
            logger.info("Reinitializing vLLM with new checkpoint...")
            del self.vllm_engine
            torch.cuda.empty_cache()
            
            self.vllm_engine = LLM(
                model=local_path,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=0.85,
                dtype="bfloat16",
                trust_remote_code=True,
            )
            logger.info("vLLM reinitialized successfully")
        except Exception:
            logger.exception("Failed to reinitialize vLLM from %s", local_path)
            # This is fatal - can't continue without generation engine
            raise

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both engines", local_path)
        return self.hf_model

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness."""
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
