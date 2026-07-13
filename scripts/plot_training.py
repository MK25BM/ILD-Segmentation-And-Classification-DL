"""
Plot training history independently (ad-hoc plotting).

Usage:
    # Plot single experiment
    python scripts/plot_training.py \
        --history artifacts/logs/20240115_143022/history.json \
        --output artifacts/plots/

    # Plot multiple experiments
    python scripts/plot_training.py \
        --history artifacts/logs/exp1/history.json \
        --history artifacts/logs/exp2/history.json \
        --output artifacts/plots/comparison/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

from core.utils import TrainingVisualizer, HistoryTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "background",
    1: "healthy",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
    7: "other_rare",
}


def main():
    parser = argparse.ArgumentParser(description="Plot training history (ad-hoc)")
    parser.add_argument(
        "--history",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to history.json file(s)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TrainingVisualizer(output_dir)
    
    logger.info(f"\nGenerating plots from {len(args.history)} history file(s)...\n")
    
    for history_file in args.history:
        logger.info(f"Processing: {history_file}")
        
        history = HistoryTracker.load(Path(history_file))
        
        experiment_name = Path(history_file).parent.name
        
        visualizer.plot_training_curves(
            history,
            save_name=f"training_curves_{experiment_name}.png",
            title=f"Training Progress ({experiment_name})"
        )
        
        visualizer.plot_class_distribution(
            history,
            class_names=CLASS_NAMES,
            save_name=f"class_performance_{experiment_name}.png"
        )
    
    logger.info(f"\n✅ All plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()

    