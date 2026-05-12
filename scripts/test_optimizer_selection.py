#!/usr/bin/env python3
"""Test the PromptOptimizer selection logic."""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reliquary.miner.prompt_optimizer import get_optimizer


class MockEnvironment:
    """Mock environment with 12500 prompts."""
    def __len__(self):
        return 12500


def main():
    optimizer = get_optimizer()
    
    if not optimizer.enabled:
        print("❌ PromptOptimizer is disabled (requests not available)")
        return 1
    
    # Create mock environment
    env = MockEnvironment()
    print(f"✅ Loaded environment with {len(env)} prompts")
    
    # Get stats
    stats = optimizer.get_stats_summary()
    print(f"📊 Database has {stats['total_prompts']} tracked prompts")
    print(f"   {stats['good_prompts']} good prompts (≥ {optimizer.min_success_rate:.0%} success rate)")
    
    # Simulate 100 prompt selections
    print("\n🎲 Simulating 100 prompt selections...")
    cooldown = set()
    selections = []
    
    for i in range(100):
        try:
            idx = optimizer.pick_prompt(env, cooldown)
            selections.append(idx)
        except Exception as e:
            print(f"❌ Error on selection {i+1}: {e}")
            break
    
    # Analyze results
    counter = Counter(selections)
    print(f"\n📈 Results:")
    print(f"  Total selections: {len(selections)}")
    print(f"  Unique prompts selected: {len(counter)}")
    
    # Check if selected prompts are in the tracked database
    tracked_count = sum(1 for idx in selections if idx in optimizer.prompt_db)
    print(f"  Selected from tracked DB: {tracked_count}/{len(selections)} ({tracked_count/len(selections)*100:.1f}%)")
    
    # Show most frequently selected
    print(f"\n🏆 Top 10 most frequently selected prompts:")
    for idx, count in counter.most_common(10):
        if idx in optimizer.prompt_db:
            pstats = optimizer.prompt_db[idx]
            print(f"  Prompt {idx:5d}: selected {count:2d} times "
                  f"(σ={pstats.avg_sigma:.3f}, {pstats.success_count}/{pstats.total_count} success)")
        else:
            print(f"  Prompt {idx:5d}: selected {count:2d} times (not in DB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
