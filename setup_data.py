# setup_data.py

import sys
import os

# Ensure we can import split_dataset from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from split_dataset import split_dataset

if __name__ == "__main__":
    split_dataset(
        source_dir="data/MIND.Funga",
        dest_dir="data/split_mind_funga",
        train_ratio=0.8
    )
