#!/usr/bin/env python3
"""Check PromptOptimizer status and scrape fresh data."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reliquary.miner.prompt_optimizer import get_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Check PromptOptimizer status")
    parser.add_argument("--scrape", action="store_true", help="Force scrape fresh data")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--top", type=int, default=20, help="Show top N prompts by success rate")
    args = parser.parse_args()
    
    optimizer = get_optimizer()
    
    if not optimizer.enabled:
        print("❌ PromptOptimizer is DISABLED (requests library not available)")
        print("\nInstall requests to enable:")
        print("  pip install requests")
        return 1
    
    print("✅ PromptOptimizer is ENABLED\n")
    
    # Scrape if requested
    if args.scrape:
        print("🔄 Scraping top miners...")
        optimizer.scrape_top_miners(force=True)
        print()
    
    # Show summary
    stats = optimizer.get_stats_summary()
    print("📊 Database Summary:")
    print(f"  Total prompts tracked: {stats['total_prompts']}")
    print(f"  Good prompts (success rate ≥ {optimizer.min_success_rate:.0%}): {stats['good_prompts']}")
    if stats['total_prompts'] > 0:
        print(f"  Average success rate: {stats['avg_success_rate']:.1%}")
    
    if stats['last_scrape_ago'] is not None:
        minutes_ago = stats['last_scrape_ago'] / 60
        print(f"  Last scraped: {minutes_ago:.1f} minutes ago")
    else:
        print(f"  Last scraped: never")
    
    print(f"\n  Exploit weight: {optimizer.exploit_weight:.0%}")
    print(f"  Scrape interval: {optimizer.scrape_interval}s")
    
    # Show top prompts
    if args.stats and optimizer.prompt_db:
        print(f"\n🏆 Top {args.top} Prompts by Success Rate:")
        print("=" * 70)
        
        # Sort by success rate
        sorted_prompts = sorted(
            [(idx, stats) for idx, stats in optimizer.prompt_db.items()],
            key=lambda x: (x[1].success_rate, x[1].total_count),
            reverse=True
        )
        
        for i, (idx, stats) in enumerate(sorted_prompts[:args.top], 1):
            if stats.total_count < 2:
                continue
            
            print(f"{i:2d}. Prompt {idx:5d}: "
                  f"{stats.success_rate:5.1%} success "
                  f"({stats.success_count}/{stats.total_count} samples, "
                  f"avg σ={stats.avg_sigma:.3f})")
    
    # Show configuration
    print("\n⚙️  Configuration:")
    print(f"  Cache file: {optimizer.cache_file}")
    print(f"  Monitoring miners:")
    for i, hotkey in enumerate(optimizer.TOP_MINERS, 1):
        print(f"    {i}. {hotkey[:8]}...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
