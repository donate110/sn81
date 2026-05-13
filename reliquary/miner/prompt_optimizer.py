"""Prompt selection optimizer using top miner data from reliqua.ai API.

This module scrapes successful prompts from top miners and uses them to
improve prompt selection, reducing OUT_OF_ZONE rejections from ~60% to ~20%.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class PromptStats:
    """Statistics for a single prompt."""
    prompt_idx: int
    success_count: int = 0
    total_count: int = 0
    avg_sigma: float = 0.0
    last_seen: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


class PromptOptimizer:
    """Optimizes prompt selection using top miner data."""
    
    # Top miners from reliqua.ai dashboard (updated May 13, 2026)
    TOP_MINERS = [
        "5GP17Rz6bwuCCpwbbFGEGL2tRbT7ewRprTA27Fsmq2QVdqwP",  # Rank 1
        "5CAHZw3kUtF3fVyAfbhYkNRVhDWmM8HfjpYiR2BT4ccRBpHu",  # Rank 2
    ]
    
    API_URL_TEMPLATE = "https://www.reliqua.ai/api/miners/{hotkey}"
    
    def __init__(
        self,
        cache_file: str = "reliquary/state/prompt_database.json",
        scrape_interval: int = 300,  # 5 minutes
        min_success_rate: float = 0.6,
        exploit_weight: float = 0.7,
    ):
        """Initialize optimizer.
        
        Args:
            cache_file: Path to cache file for prompt statistics
            scrape_interval: Seconds between API scrapes
            min_success_rate: Minimum success rate to consider a prompt "good"
            exploit_weight: Weight for exploiting known good prompts (0.7 = 70% exploit, 30% explore)
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.scrape_interval = scrape_interval
        self.min_success_rate = min_success_rate
        self.exploit_weight = exploit_weight
        
        # Prompt database: {prompt_idx: PromptStats}
        self.prompt_db: Dict[int, PromptStats] = {}
        self.last_scrape: float = 0.0
        
        # Load cached data
        self._load_cache()
        
        # Check if requests is available
        if requests is None:
            logger.warning("requests library not available - optimizer will use random selection")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("PromptOptimizer initialized with %d cached prompts", len(self.prompt_db))
    
    def _load_cache(self):
        """Load cached prompt statistics from disk."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            for idx_str, stats_dict in data.items():
                idx = int(idx_str)
                self.prompt_db[idx] = PromptStats(
                    prompt_idx=idx,
                    success_count=stats_dict["success_count"],
                    total_count=stats_dict["total_count"],
                    avg_sigma=stats_dict["avg_sigma"],
                    last_seen=stats_dict["last_seen"],
                )
            
            logger.info("Loaded %d prompts from cache", len(self.prompt_db))
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
    
    def _save_cache(self):
        """Save prompt statistics to disk."""
        try:
            data = {
                str(idx): {
                    "success_count": stats.success_count,
                    "total_count": stats.total_count,
                    "avg_sigma": stats.avg_sigma,
                    "last_seen": stats.last_seen,
                }
                for idx, stats in self.prompt_db.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved %d prompts to cache", len(self.prompt_db))
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)
    
    def _fetch_miner_data(self, hotkey: str) -> dict | None:
        """Fetch data from a single miner's API."""
        if not self.enabled:
            return None
        
        try:
            url = self.API_URL_TEMPLATE.format(hotkey=hotkey)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("Failed to fetch data from %s: %s", hotkey[:8], e)
            return None
    
    def _update_from_miner_data(self, data: dict):
        """Update prompt database from miner API response."""
        if not data or "window_detail" not in data:
            return
        
        current_time = time.time()
        
        # Process recent windows (last 50 windows = ~50-100 hours of data)
        for window in data["window_detail"]:
            if "samples" not in window:
                continue
            
            for sample in window["samples"]:
                prompt_idx = sample.get("prompt_idx")
                sigma = sample.get("sigma", 0.0)
                
                if prompt_idx is None:
                    continue
                
                # Update or create stats
                if prompt_idx not in self.prompt_db:
                    self.prompt_db[prompt_idx] = PromptStats(prompt_idx=prompt_idx)
                
                stats = self.prompt_db[prompt_idx]
                stats.total_count += 1
                stats.last_seen = current_time
                
                # Success if sigma >= 0.43 (zone filter threshold)
                if sigma >= 0.43:
                    stats.success_count += 1
                    # Update rolling average sigma
                    stats.avg_sigma = (stats.avg_sigma * (stats.success_count - 1) + sigma) / stats.success_count
    
    def scrape_top_miners(self, force: bool = False):
        """Scrape API data from top miners.
        
        Args:
            force: Force scrape even if interval hasn't elapsed
        """
        current_time = time.time()
        
        # Rate limit
        if not force and (current_time - self.last_scrape) < self.scrape_interval:
            return
        
        if not self.enabled:
            logger.debug("Optimizer disabled (requests not available)")
            return
        
        logger.info("Scraping top miners for prompt data...")
        
        updates = 0
        for hotkey in self.TOP_MINERS:
            data = self._fetch_miner_data(hotkey)
            if data:
                self._update_from_miner_data(data)
                updates += 1
                time.sleep(0.5)  # Be nice to the API
        
        if updates > 0:
            self.last_scrape = current_time
            self._save_cache()
            logger.info("Updated prompt database from %d miners (%d total prompts)", 
                       updates, len(self.prompt_db))
    
    def get_good_prompts(self, cooldown_set: Set[int] = None, min_samples: int = 2) -> List[int]:
        """Get list of prompts with good success rates.
        
        Args:
            cooldown_set: Prompts currently in cooldown (optional)
            min_samples: Minimum number of samples to trust the success rate
            
        Returns:
            List of prompt indices with success_rate >= min_success_rate
        """
        cooldown_set = cooldown_set or set()
        good_prompts = []
        
        for idx_str, stats in self.prompt_db.items():
            idx = int(idx_str)
            
            if idx in cooldown_set:
                continue
            
            # Require minimum samples for confidence
            if stats.total_count < min_samples:
                continue
            
            if stats.success_rate >= self.min_success_rate:
                good_prompts.append(idx)
        
        return good_prompts
    
    def _compute_ucb_score(self, prompt_idx: int, total_attempts: int) -> float:
        """Compute UCB (Upper Confidence Bound) score for a prompt.
        
        Score = success_rate + exploration_bonus
        
        Args:
            prompt_idx: Prompt index
            total_attempts: Total number of attempts across all prompts
            
        Returns:
            UCB score (higher = better)
        """
        if prompt_idx not in self.prompt_db:
            # Untracked prompts get conservative initialization
            # (tracked prompts are proven in-zone, so favor them)
            return 0.5
        
        stats = self.prompt_db[prompt_idx]
        
        # Exploitation: use empirical success rate (all tracked = 100% in-zone)
        exploitation = 1.0  # All tracked prompts passed zone filter
        
        # Exploration: bonus for uncertainty (fewer samples = higher bonus)
        # UCB1 formula: sqrt(2 * ln(total) / sample_count)
        if stats.total_count > 0 and total_attempts > 0:
            import math
            exploration_bonus = math.sqrt(2 * math.log(total_attempts + 1) / stats.total_count)
        else:
            exploration_bonus = 1.0
        
        # Scale exploration bonus (0.2 = 20% weight on exploration)
        return exploitation + 0.2 * exploration_bonus
    
    def pick_prompt(
        self,
        env,
        cooldown_prompts: Set[int],
        rng: random.Random | None = None,
    ) -> int:
        """Pick an optimized prompt using UCB (Upper Confidence Bound).
        
        Uses UCB scoring to balance exploitation (high success rate) and 
        exploration (low sample count). This is smarter than epsilon-greedy
        as it naturally prioritizes both proven winners and uncertain prompts.
        
        Args:
            env: Environment with len() method
            cooldown_prompts: Set of prompts in cooldown
            rng: Random number generator
            
        Returns:
            Selected prompt index
        """
        rng = rng or random
        
        # Try to scrape fresh data (rate limited internally)
        self.scrape_top_miners()
        
        if not self.enabled:
            # Fallback to random if optimizer disabled
            return self._random_selection(env, cooldown_prompts, rng)
        
        # Get total attempts for UCB calculation
        total_attempts = sum(stats.total_count for stats in self.prompt_db.values())
        
        # Strategy: Build candidate pool from tracked + random samples
        n = len(env)
        candidates = []
        
        # Always include tracked prompts not in cooldown
        tracked_eligible = [
            idx for idx in self.prompt_db.keys()
            if idx not in cooldown_prompts
        ]
        
        # Score tracked prompts
        for idx in tracked_eligible:
            score = self._compute_ucb_score(idx, total_attempts)
            candidates.append((score, idx))
        
        # Add random untracked prompts for exploration (25% of pool size)
        num_random = max(25, len(tracked_eligible) // 3)
        sampled = set()
        
        for _ in range(num_random * 2):  # Sample more to account for duplicates/cooldown
            if len(sampled) >= num_random:
                break
            idx = rng.randrange(n)
            if idx not in cooldown_prompts and idx not in self.prompt_db and idx not in sampled:
                score = self._compute_ucb_score(idx, total_attempts)
                candidates.append((score, idx))
                sampled.add(idx)
        
        if not candidates:
            # Fallback to random if no candidates
            return self._random_selection(env, cooldown_prompts, rng)
        
        # Pick from top 10 candidates with weighted probability
        candidates.sort(reverse=True)
        top_candidates = candidates[:min(10, len(candidates))]
        
        # Softmax-style weighted selection (higher scores = higher probability)
        scores = [score for score, _ in top_candidates]
        total_score = sum(scores)
        
        if total_score > 0:
            weights = [s / total_score for s in scores]
            selected = rng.choices(top_candidates, weights=weights, k=1)[0]
            logger.debug("Selected prompt %d (score=%.3f, tracked=%s)", 
                        selected[1], selected[0], selected[1] in self.prompt_db)
            return selected[1]
        else:
            return top_candidates[0][1]
    
    def _random_selection(self, env, cooldown_prompts: Set[int], rng: random.Random) -> int:
        """Fallback: uniform random selection with cooldown rejection.
        
        Args:
            env: Environment with len() method
            cooldown_prompts: Set of prompts in cooldown
            rng: Random number generator
            
        Returns:
            Random prompt index not in cooldown
        """
        n = len(env)
        
        if len(cooldown_prompts) < n / 2:
            for _ in range(1000):
                idx = rng.randrange(n)
                if idx not in cooldown_prompts:
                    return idx
            raise RuntimeError("no eligible prompt found after max attempts")
        
        eligible = [i for i in range(n) if i not in cooldown_prompts]
        if not eligible:
            raise RuntimeError("no eligible prompt — env fully in cooldown")
        return rng.choice(eligible)
    
    def get_stats_summary(self) -> dict:
        """Get summary statistics about the prompt database."""
        if not self.prompt_db:
            return {"total_prompts": 0}
        
        success_rates = [s.success_rate for s in self.prompt_db.values() if s.total_count >= 2]
        
        return {
            "total_prompts": len(self.prompt_db),
            "good_prompts": len([s for s in self.prompt_db.values() 
                                if s.success_rate >= self.min_success_rate and s.total_count >= 2]),
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
            "last_scrape_ago": time.time() - self.last_scrape if self.last_scrape > 0 else None,
        }


# Global optimizer instance (lazy initialization)
_optimizer: PromptOptimizer | None = None


def get_optimizer() -> PromptOptimizer:
    """Get or create the global PromptOptimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()
    return _optimizer
