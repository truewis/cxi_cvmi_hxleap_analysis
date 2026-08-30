#!/usr/bin/env python3
"""
Append a "## Total shots per (run, step)" section to the NOTES.md of every
aggregate_4roi_* folder under batch_analysis/.

For each (run, step, kind) pair, "total shots" is the length of the
`data_peak_positions_run_<N>.npy['x']` array in the source
circular_wiggler_<kind>_run<N>_step<M>_batch_metrics/ directory. That is
the number of shots the wiggle analysis pipelined for that (run, step) —
the base pool before ROI selection.

Sources:
  * _by_energy folders → pairs from aggregation_metadata.npy.
  * _steps / run145_* folders → pairs parsed from NOTES.md.
  * compare folder → union of its two sibling folders' pairs.

The section is added ONCE per folder: if a "## Total shots per (run, step)"
heading already exists, the script leaves the file alone.

Requires conda env CXI.
"""
import argparse
import os
import re

import numpy as np


BATCH_ANALYSIS_DIR = '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/batch_analysis_results/legacy_outputs'
BATCH_RESULTS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                     'batch_analysis_results/results_145_to_152_4_ROI')


def _peak_positions_path(run, step, kind):
    return os.path.join(
        BATCH_RESULTS_ROOT,
        f'circular_wiggler_{kind}_run{run}_step{step}_batch_metrics',
        f'data_peak_positions_run_{run}.npy',
    )


def _shots_for_pair(run, step, kind):
    """Return int shot count for (run, step, kind), or None if the source
    peak-positions file is missing."""
    p = _peak_positions_path(run, step, kind)
    if not os.path.exists(p):
        return None
    obj = np.load(p, allow_pickle=True).item()
    x = obj.get('x')
    if x is None:
        return None
    return int(len(x))


def _pairs_from_steps_notes(notes_path):
    """Parse the pair list from a `_steps`-style NOTES.md. Handles the two
    formats we emit:
      * preset-specific tables (Requested / Available / Substitution)
      * custom-list code block with 'run:step, ...'
    Returns the union of all pairs that appear in any ROI or Total table."""
    text = open(notes_path).read()
    pairs = set()
    # Rows like "| 145 | 14 | ..."
    for m in re.finditer(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', text, re.MULTILINE):
        pairs.add((int(m.group(1)), int(m.group(2))))
    # Code-block "145:14, 145:15, ..."
    for m in re.finditer(r'(\d+)\s*:\s*(\d+)', text):
        pairs.add((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


def _kind_from_folder_name(name):
    if '_goose_' in name:
        return 'goose'
    if '_duck_' in name:
        return 'duck'
    return 'duck'  # sensible default


def _resolve_pairs_and_kind(folder):
    """Return (pairs, kind) or (None, None) if we can't resolve them."""
    name = os.path.basename(folder)
    kind = _kind_from_folder_name(name)

    meta_path = os.path.join(folder, 'aggregation_metadata.npy')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True).item()
        pairs = [tuple(p) for p in meta.get('pairs', [])]
        meta_kind = meta.get('kind')
        if meta_kind:
            kind = meta_kind
        return sorted(set(pairs)), kind

    notes_path = os.path.join(folder, 'NOTES.md')
    if not os.path.exists(notes_path):
        return None, None

    # Compare folder: union of the two sibling folders' pairs.
    if 'compare_' in name:
        parent = os.path.dirname(folder)
        siblings = []
        for m in re.finditer(r'`(aggregated_4roi_\w+)/', open(notes_path).read()):
            sib = os.path.join(parent, m.group(1))
            if os.path.isdir(sib):
                siblings.append(sib)
        if not siblings:
            return None, kind
        pairs = set()
        for sib in siblings:
            sp, sk = _resolve_pairs_and_kind(sib)
            if sp:
                pairs.update(sp)
                kind = sk or kind
        return sorted(pairs), kind

    return _pairs_from_steps_notes(notes_path), kind


TOTAL_SECTION_HEADER = '## Total shots per (run, step)'


def already_has_section(notes_path):
    if not os.path.exists(notes_path):
        return False
    return TOTAL_SECTION_HEADER in open(notes_path).read()


def build_section(pairs, kind):
    counts = []
    missing = []
    for run, step in pairs:
        n = _shots_for_pair(run, step, kind)
        if n is None:
            missing.append((run, step))
            continue
        counts.append((run, step, n))
    lines = ['', TOTAL_SECTION_HEADER, '',
             f'kind = {kind}. "Total shots" is the length of '
             '`data_peak_positions_run_<N>.npy[\'x\']` in the source '
             '`circular_wiggler_<kind>_run<N>_step<M>_batch_metrics/` '
             'directory — the number of shots pipelined into the wiggle '
             'analysis for that (run, step), before ROI selection.',
             '']
    lines.append('| run | step | total shots |')
    lines.append('| --- | --- | --- |')
    grand = 0
    for r, s, n in counts:
        lines.append(f'| {r} | {s} | {n} |')
        grand += n
    lines.append(f'| **TOTAL** | | **{grand}** |')
    lines.append('')
    if missing:
        lines.append('Missing source files (no total-shot count available):')
        for r, s in missing:
            lines.append(f'- run {r}, step {s}')
        lines.append('')
    return '\n'.join(lines)


def process(folder):
    name = os.path.basename(folder)
    notes = os.path.join(folder, 'NOTES.md')
    if not os.path.exists(notes):
        print(f'  [skip] {name}: no NOTES.md')
        return
    if already_has_section(notes):
        print(f'  [skip] {name}: section already present')
        return
    pairs, kind = _resolve_pairs_and_kind(folder)
    if not pairs:
        print(f'  [skip] {name}: could not resolve pair list')
        return

    section = build_section(pairs, kind)
    with open(notes, 'a') as f:
        f.write(section)
    print(f'  appended to {notes}  ({len(pairs)} pairs, kind={kind})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='substring; only folders whose basename contains this run')
    args = ap.parse_args()

    folders = sorted(
        os.path.join(BATCH_ANALYSIS_DIR, d)
        for d in os.listdir(BATCH_ANALYSIS_DIR)
        if d.startswith('aggregated_4roi_')
    )
    if args.only:
        folders = [f for f in folders if args.only in os.path.basename(f)]

    for f in folders:
        process(f)


if __name__ == '__main__':
    main()
