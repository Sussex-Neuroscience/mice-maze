"""
kpms_split_bouts.py

Trim keypoint-MoSeq input down to the frames where the mouse is on the maze.

Why this exists
---------------
`filter_in_maze.py` leaves every frame without a real mouse as all-NaN. In a
typical session that is 50-70% of the recording (the mouse is off the maze
between trials). keypoint-MoSeq then warns

    Coordinates for the following bodyparts are missing (set to NaN) in at
    least 50.0% of frames

and, worse, interpolates straight through those gaps, so minutes of "no mouse"
become a frozen pose that the model learns as a fake behaviour. While the mouse
is present the tracking is good (midbody/tailbase ~0% missing, nose/endtail
~10%), so the fix is to drop the absent stretches, not to fill them.

Do NOT replace NaN with 0: keypoint-MoSeq would treat (0, 0) as a real position
in the image corner.

What it does
------------
Splits each recording into "bouts": runs of frames where the anchor bodypart is
tracked. Gaps shorter than `max_gap` frames are kept (kpms interpolates those),
and bouts shorter than `min_len` frames are discarded. Each bout becomes its own
recording named `<recording>_f<start_frame>` so syllables can be mapped back to
the original video frame.

Usage, in the keypoint-MoSeq notebook right after loading:

    import sys; sys.path.insert(0, r"<repo>\\analysis")
    from kpms_split_bouts import split_bouts

    coordinates, confidences, bodyparts = kpms.load_keypoints(data_dir, "sleap")
    coordinates, confidences = split_bouts(coordinates, confidences, bodyparts)
    data, metadata = kpms.format_data(coordinates, confidences, **config())

Only needs numpy, so it runs in the keypoint-MoSeq environment as-is.
"""

import numpy as np

DEFAULT_ANCHORS = ("midbody", "tailbase")  # most reliably tracked nodes
DEFAULT_MAX_GAP = 15    # frames (~0.5 s at 30 fps) of dropout bridged inside a bout
DEFAULT_MIN_LEN = 90    # frames (~3 s at 30 fps); shorter bouts are discarded


def present_mask(coords, bodyparts, anchors=DEFAULT_ANCHORS):
    """True for frames where any anchor bodypart has coordinates.
    `coords` is (T, K, 2) as returned by kpms.load_keypoints."""
    idx = [bodyparts.index(a) for a in anchors if a in bodyparts]
    if not idx:
        raise ValueError(f"none of {anchors} are in bodyparts {bodyparts}")
    return ~np.isnan(coords[:, idx, 0]).all(axis=1)


def find_bouts(present, max_gap=DEFAULT_MAX_GAP, min_len=DEFAULT_MIN_LEN):
    """Return [(start, stop), ...] half-open frame ranges of presence, with
    gaps of <= max_gap frames merged and bouts < min_len frames dropped."""
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    stops = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_len]


def split_bouts(coordinates, confidences, bodyparts, anchors=DEFAULT_ANCHORS,
                max_gap=DEFAULT_MAX_GAP, min_len=DEFAULT_MIN_LEN, verbose=True):
    """Split kpms coordinate/confidence dicts into per-bout recordings.

    Returns new (coordinates, confidences) dicts keyed `<name>_f<start>`."""
    out_coords, out_confs = {}, {}
    for name, coords in coordinates.items():
        conf = confidences[name] if confidences is not None else None
        bouts = find_bouts(present_mask(coords, bodyparts, anchors),
                           max_gap, min_len)
        for a, b in bouts:
            key = f"{name}_f{a}"
            # float64: kpms runs JAX in 64-bit mode and rejects float32 data
            out_coords[key] = coords[a:b].astype(np.float64)
            if conf is not None:
                out_confs[key] = conf[a:b].astype(np.float64)
        if verbose:
            kept = sum(b - a for a, b in bouts)
            print(f"{name}: {len(bouts)} bouts, kept {kept}/{len(coords)} frames "
                  f"({100.0 * kept / max(len(coords), 1):.0f}%)")
    return out_coords, (out_confs if confidences is not None else None)
