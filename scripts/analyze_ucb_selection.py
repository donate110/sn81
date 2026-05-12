#!/usr/bin/env python3
"""Detailed analysis of UCB prompt selection behavior."""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from reliquary.miner.prompt_optimizer import get_optimizer


class MockEnv:
    """Mock environment with 12500 prompts."""
    def __len__(self):
        return 12500


def main():
    optimizer = get_optimizer()
    
    if not optimizer.enabled:
        print("❌ PromptOptimizer is disabled")
        return 1
    
    env = MockEnv()
    cooldown = set()
    
    print("📊 UCB Selection Analysis")
    print("=" * 70)
    
    # Get current DB stats
    stats_summary = optimizer.get_stats_summary()
    total_attempts = sum(s.total_count for s in optimizer.prompt_db.values())
    
    print(f"\n📂 Database Status:")
    print(f"  Tracked prompts: {stats_summary['total_prompts']}")
    print(f"  Total attempts recorded: {total_attempts}")
    
    # Show score distribution
    print(f"\n🎯 Top 20 Prompts by UCB Score:")
    print("=" * 70)
    
    scored_prompts = []
    for idx, stats in optimizer.prompt_db.items():
        score = optimizer._compute_ucb_score(idx, total_attempts)
        scored_prompts.append((score, idx, stats))
    
    scored_prompts.sort(reverse=True)
    
    for i, (score, idx, stats) in enumerate(scored_prompts[:20], 1):
        print(f"{i:2d}. Prompt {idx:5d}: score={score:.4f} "
              f"(σ={stats.avg_sigma:.3f}, {stats.success_count}/{stats.total_count} samples)")
    
    # Simulate selections and track distribution
    print(f"\n🎲 Simulating 500 selections...")
    selections = []
    
    for _ in range(500):
        idx = optimizer.pick_prompt(env, cooldown)
        selections.append(idx)
    
    counter = Counter(selections)
    
    print(f"\n📈 Selection Results:")
    print(f"  Total selections: {len(selections)}")
    print(f"  Unique prompts: {len(counter)}")
    
    tracked = sum(1 for idx in selections if idx in optimizer.prompt_db)
    print(f"  From tracked DB: {tracked}/{len(selections)} ({tracked/len(selections)*100:.1f}%)")
    
    # Analyze sample count distribution of selected prompts
    sample_counts = []
    for idx in selections:
        if idx in optimizer.prompt_db:
            sample_counts.append(optimizer.prompt_db[idx].total_count)
    
    if sample_counts:
        avg_samples = sum(sample_counts) / len(sample_counts)
        print(f"  Average sample count of selections: {avg_samples:.1f}")
    
    print(f"\n🔥 Top 15 Most Selected Prompts:")
    print("=" * 70)
    
    for i, (idx, count) in enumerate(counter.most_common(15), 1):
        if idx in optimizer.prompt_db:
            stats = optimizer.prompt_db[idx]
            score = optimizer._compute_ucb_score(idx, total_attempts)
            print(f"{i:2d}. Prompt {idx:5d}: selected {count:3d} times "
                  f"(score={score:.4f}, {stats.success_count}/{stats.total_count} samples, σ={stats.avg_sigma:.3f})")
        else:
            print(f"{i:2d}. Prompt {idx:5d}: selected {count:3d} times (UNTRACKED)")
    
    # Check exploration
    untracked = sum(1 for idx in selections if idx not in optimizer.prompt_db)
    print(f"\n🔍 Exploration Metrics:")
    print(f"  Untracked prompts selected: {untracked}/{len(selections)} ({untracked/len(selections)*100:.1f}%)")
    print(f"  Untracked prompts discovered: {len([idx for idx in counter if idx not in optimizer.prompt_db])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
