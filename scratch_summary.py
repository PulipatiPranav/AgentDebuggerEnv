import json
import glob
import os
import numpy as np
from collections import defaultdict

def extract_solve_rate(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data['overall']['solve_rate']

def main():
    arms = ['E1', 'E3', 'E4']
    splits = {
        'heldout': 'results/primary/{arm}_s{seed}.json',
        'train': 'results/diagnostics/{arm}_s{seed}_train_eval.json',
        'quixbugs': 'results/diagnostics/{arm}_s{seed}_quixbugs_eval.json'
    }
    
    for arm in arms:
        print(f"--- ARM: {arm} ---")
        for split, pattern in splits.items():
            rates = []
            for seed in [42, 123, 456]:
                path = pattern.format(arm=arm, seed=seed)
                if os.path.exists(path):
                    rates.append(extract_solve_rate(path))
            
            if rates:
                mean_rate = np.mean(rates) * 100
                std_rate = np.std(rates) * 100
                print(f"  {split.ljust(10)}: {mean_rate:.1f}% ± {std_rate:.1f}%")

if __name__ == '__main__':
    main()
