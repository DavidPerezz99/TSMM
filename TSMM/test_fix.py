#!/usr/bin/env python
"""Quick test to verify the nested config fix."""

import sys
import yaml
from hypersearch import generate_smart_experiments

try:
    # Load configs
    with open('config/config.yaml') as f:
        base_cfg = yaml.safe_load(f)
    
    with open('config/sweep_definition.yaml') as f:
        sweep_cfg = yaml.safe_load(f)
    
    print("Loading configs...")
    print(f"Base config loaded: {type(base_cfg)}")
    print(f"Sweep config loaded: {type(sweep_cfg)}")
    
    # Generate smart experiments
    print("\nGenerating smart experiments...")
    exp_count = 0
    for exp in generate_smart_experiments(base_cfg, sweep_cfg):
        exp_count += 1
        if exp_count <= 2:
            print(f"\nExperiment {exp_count}:")
            # Show nbeats config if present
            if 'nbeats' in exp:
                print(f"  nbeats keys: {list(exp['nbeats'].keys())}")
                if 'stacks_config' in exp['nbeats']:
                    print(f"  stacks_config type: {type(exp['nbeats']['stacks_config'])}")
                    print(f"  stacks_config: {exp['nbeats']['stacks_config']}")
    
    print(f"\n✓ SUCCESS: Generated {exp_count} experiments without error!")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
