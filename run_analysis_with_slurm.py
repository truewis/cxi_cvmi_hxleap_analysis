#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from xleap_analysis import compute_circular_wiggle_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Multi-Mask Step Parser Node")
    parser.add_argument("--pickle_file", type=str, required=True, help="Path to preprocessed pkl file")
    parser.add_argument("--mask", type=str, choices=["goose", "duck", "gasjet"], required=True)
    
    # Adding an optional steps parameter. 'nargs="+"' allows passing space-separated integers (e.g., --steps 0 1 4)
    parser.add_argument("--steps", type=int, nargs="+", default=None, 
                        help="Specific step indices to process. If omitted, all steps are analyzed.")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.pickle_file}...")
    with open(args.pickle_file, "rb") as f:
        data = pickle.load(f)
        
    # --- STEP DETERMINATION LOGIC ---
    # Check if 'step' or 'step_indices' array exists in your pickle data. 
    # (Assuming you saved a 'step' tracking array corresponding to each shot)
    if args.steps is not None:
        target_steps = args.steps
        print(f"Processing explicitly specified steps: {target_steps}")
    else:
        # Automatically find all unique steps present in your preprocessed data
        if "step" in data:
            target_steps = sorted(list(np.unique(data["step"])))
            print(f"No steps specified. Found all available steps automatically: {target_steps}")
        else:
            # Fallback if your data doesn't use step slices
            target_steps = [0]
            print("No step array found in data; defaulting to a single step execution.")

    # --- LOOP THROUGH SELECTED STEPS ---
    for step in target_steps:
        print(f"\n--- Starting Analysis for Step {step} ---")
        
        # Slice your mask array to only process shots belonging to this specific step
        # (Change `data["step"] == step` to match your actual step-tracking variable)
        if "step" in data:
            step_mask_condition = (data["step"] == step)
            combined_mask = data["masks"][args.mask] & step_mask_condition
        else:
            combined_mask = data["masks"][args.mask]

        # Skip if no valid events exist for this mask + step combination
        if not np.any(combined_mask):
            print(f"No matching shots found for mask '{args.mask}' in Step {step}. Skipping.")
            continue

        compute_circular_wiggle_analysis(
            mask_array=combined_mask,
            run_id=data["run"],
            output_dir_suffix=f"{args.mask}_step_{step}",
            images=data["images"],
            hits=data["hits"],
            mean_energy=data["mean_energy"],
            is_gaussian=data["is_gaussian"],
            total_hit_within_mask=data["total_hit_within_mask"],
            original_event_number=data["original_event_number"],
            annulus_mask=data["annulus_mask"]
        )
