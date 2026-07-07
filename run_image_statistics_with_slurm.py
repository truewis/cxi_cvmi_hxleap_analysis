#!/usr/bin/env python3
"""
SLURM entry point: run the multi-method image-statistics comparison of
streaked vs unstreaked shots per LXT step.

By convention goose = unstreaked control, duck = streaked test — pass
--swap to invert.

Outputs land in `image_statistics_<mask>_run<run>_step<N>_batch_metrics/`.
"""
import argparse
import pickle

import numpy as np

from analysis_library.image_statistics_analysis import compute_image_statistics_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-method image-statistics comparison of streaked (duck) "
                    "vs unstreaked (goose) shots, per LXT step."
    )
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Path to preprocessed pickle produced by the run notebook")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="LXT step indices to process (default: all)")
    parser.add_argument("--r_adjustment", type=float, default=0.0,
                        help="Additive adjustment to re for the composite score map")
    parser.add_argument("--swap", action="store_true",
                        help="Use duck as control and goose as test instead")
    args = parser.parse_args()

    print(f"Loading data from {args.pickle_file}...")
    with open(args.pickle_file, "rb") as f:
        data = pickle.load(f)

    if "lxts" in data:
        step_axis_name = "lxts"
        unique_steps = np.unique(data["lxts"])
    elif "step" in data:
        step_axis_name = "step"
        unique_steps = np.unique(data["step"])
    else:
        step_axis_name = None
        unique_steps = np.array([0])

    if args.steps is not None:
        steps_to_process = args.steps
    else:
        steps_to_process = np.arange(len(unique_steps))

    control_mask_name, test_mask_name = ("goose", "duck") if not args.swap else ("duck", "goose")
    control_prefix, test_prefix = control_mask_name, test_mask_name

    is_gaussian = data["is_gaussian"].astype(bool)
    control_mask_all = data["masks"][control_mask_name].astype(bool) & is_gaussian
    test_mask_all    = data["masks"][test_mask_name].astype(bool)  & is_gaussian

    for target_step in steps_to_process:
        if target_step >= len(unique_steps):
            print(f"Requested step {target_step} out of bounds. Skipping.")
            continue

        if step_axis_name is not None:
            step_val = unique_steps[target_step]
            step_condition = (data[step_axis_name] == step_val)
        else:
            step_condition = np.ones(len(is_gaussian), dtype=bool)

        control_mask = control_mask_all & step_condition
        test_mask    = test_mask_all    & step_condition

        print(f"\nStep index={target_step}  | control={control_mask.sum()}  | test={test_mask.sum()}")

        compute_image_statistics_analysis(
            control_mask_array=control_mask,
            test_mask_array=test_mask,
            run_id=data["run"],
            target_step=target_step,
            images=data["images"],
            hits=data["hits"],
            mean_energy=data["mean_energy"],
            total_hit_within_mask=data["total_hit_within_mask"],
            original_event_number=data["original_event_number"],
            annulus_mask=data.get("annulus_mask", None),
            r_adjustment=args.r_adjustment,
            output_dir_suffix=f"{test_prefix}vs{control_prefix}_run{data['run']}_step{target_step}",
            control_prefix=control_prefix,
            test_prefix=test_prefix,
        )
