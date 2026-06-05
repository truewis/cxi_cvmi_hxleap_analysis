#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from analysis_library.cvmi import compute_circular_wiggle_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Multi-Mask Summary Diagnostic Pipeline")
    parser.add_argument("--pickle_file", type=str, required=True, help="Path to pkl data pack")
    parser.add_argument("--mask", type=str, choices=["goose", "duck"], required=True)
    parser.add_argument("--steps", type=int, nargs="+", help="Steps to process")
    parser.add_argument("--r_adjustment", type=float, help="Adjustment in scoring radius")
    args = parser.parse_args()
    
    print(f"Loading data from {args.pickle_file}...")
    with open(args.pickle_file, "rb") as f:
        data = pickle.load(f)
        
    # Determine distinct unique step boundaries present in dataset
    if "lxts" in data:
        unique_steps = np.unique(data["lxts"])
    elif "step" in data:
        unique_steps = np.unique(data["step"])
    else:
        unique_steps = [0]

    if args.steps is not None:
        steps_to_process = args.steps
    else:
        steps_to_process = np.arange(len(unique_steps))

    if args.r_adjustment is not None:
        r_adjustment = args.r_adjustment
    else:
        r_adjustment = 0
    
    for target_step in steps_to_process:
        if target_step >= len(unique_steps):
            print(f"Requested step {target_step} is out of bounds. Skipping.")
            continue
            
        print(f"\nEvaluating Context: Mask={args.mask} | Step index={target_step}")
        
        # Build composite step constraint slice boolean array
        step_val = unique_steps[target_step]
        step_mask_condition = (data["lxts"] == step_val) if "lxts" in data else (data["step"] == step_val)
        
        combined_mask = step_mask_condition & data["masks"][args.mask].astype(bool) & data["is_gaussian"]
        
        # Call the updated library function containing all plots
        compute_circular_wiggle_analysis(
            mask_array=combined_mask,
            run_id=data["run"],
            target_step=target_step,
            images=data["images"],
            hits=data["hits"],
            mean_energy=data["mean_energy"],
            is_gaussian=data["is_gaussian"],
            total_hit_within_mask=data["total_hit_within_mask"],
            original_event_number=data["original_event_number"],
            annulus_mask=data["annulus_mask"],
            output_prefix=args.mask,
            r_adjustment = r_adjustment
        )
