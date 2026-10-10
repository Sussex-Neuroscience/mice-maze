#!/usr/bin/env python3
"""Survey the standalone sessions: which mice, which maze, which stimuli.

    python analysis/auditory/survey_standalone_sessions.py
    python analysis/auditory/survey_standalone_sessions.py --root D:\\AuditoryExperiments

The sessions recorded before the eight-arm protocol sit loose at the top of the
data folder, one folder per session, beside the 8_arms*, grammar and dopamine
folders that hold the later experiments. The code that wrote them changed
several times, so their trials files come in six layouts: pure tones, tone
sequences, two versions of musical intervals, and so on. This reads each file
on its own terms and reports, for every session:

    the mouse, its sex and birth date, from the per-mouse file
    when the session ran, from the timestamps inside the trials file
    how many arms the maze had, from the regions of interest logged
    what was played, in which block, on which arm
    how long the animal spent in each arm, and how many visits it made

One mouse on one day is one session, however many files it took. When a
recording crashed and the session was restarted, each restart wrote its own
trials file; those are merged into one session, and every block keeps a note of
which part it came from, so the parts can still be told apart. A file saved
twice with the same contents is a single recording, not a restart, and is
counted once, with both file names listed.

It writes two tables, and touches nothing in the data folder:

    sessions.csv   one row per session (one mouse, one day)
    blocks.csv     one row per arm per block: the stimulus it played, the time
                   spent there and the visits, which is what an occupancy
                   analysis starts from

By default they go in a folder beside the data, D:\\AuditoryExperiments_survey,
so derived files are never mixed in with the recordings.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

ROOT = Path(r"D:\AuditoryExperiments")

# Folders that hold the later experiments, not standalone sessions.
LATER = re.compile(r"^(8_arms|grammar|dopamine|_)", re.I)

# How the different versions of the code wrote "nothing playing here". In the
# interval sessions an arm marked no_interval played [0, 0]: a silent arm
# inside a sound block, not a block of silence.
SILENT = {"", "0", "0.0", "none", "nan", "silent", "[]", "[0]", "no_interval"}


# -- small readers -----------------------------------------------------------
def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def pick(row: Dict[str, str], *names: str) -> Optional[str]:
    """The first of several possible column names that the row has."""
    for name in names:
        if name in row and row[name] is not None:
            return row[name].strip()
    return None


def as_float(text: Optional[str]) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def numbers_in(text: Optional[str]) -> List[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def read_mouse(folder: Path) -> Dict[str, str]:
    """The per-mouse file: ID, ear mark, birth date and sex."""
    for f in sorted(folder.glob("*.csv")):
        if f.name.lower().startswith("trials"):
            continue
        rows = read_rows(f)
        if rows and "animal ID" in rows[0]:
            r = rows[0]
            return {"ear_mark": r.get("ear mark identifier", "").strip(),
                    "birth_date": r.get("animal birth date", "").strip(),
                    "sex": r.get("animal gender", "").strip().lower(),
                    "mouse_file_id": r.get("animal ID", "").strip()}
    return {"ear_mark": "", "birth_date": "", "sex": "", "mouse_file_id": ""}


# -- what was played ----------------------------------------------------------
def stimulus_kind(columns: List[str]) -> str:
    cols = set(columns)
    if "pattern" in cols:
        return "sequences"
    if "interval" in cols:
        return "intervals"
    if "frequency" in cols:
        return "pure tones"
    return "unknown"


def sequence_name(text: str) -> str:
    """Name a logged sequence by the unit it repeats: AoAo, ABAB, ABCABC.

    The sequences were logged in full, as a list of symbols ('o' is a silent
    slot), so a 200-symbol ABAB... reads as a wall of letters. A sequence with
    no short repeating unit was the random one, which differed every block.
    """
    try:
        symbols = "".join(ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError):
        symbols = re.sub(r"[^A-Za-z]", "", text)
    if not symbols:
        return "silent"
    for k in range(1, 5):
        unit = symbols[:k]
        if (unit * (len(symbols) // k + 1))[:len(symbols)] == symbols:
            return unit * 2
    return "random"


def sequence_tones(row: Dict[str, str]) -> str:
    """Which frequency each letter of a sequence stood for, e.g. 'A=1000;B=7000'.

    The sequence sessions log the pattern and its frequencies slot by slot, so
    the two lists can be read side by side. The letters meant different tones
    in different sessions, so this is recorded per row rather than assumed.
    """
    try:
        symbols = ast.literal_eval(pick(row, "pattern") or "")
        freqs = ast.literal_eval(pick(row, "frequency_seq") or "")
    except (ValueError, SyntaxError, TypeError):
        return ""
    if not isinstance(symbols, (list, tuple)) or not isinstance(freqs, (list, tuple)):
        return ""
    tone_of = {}
    for s, f in zip(symbols, freqs):
        if s != "o" and float(f) > 0:
            tone_of.setdefault(str(s), float(f))
    return ";".join(f"{k}={v:g}" for k, v in sorted(tone_of.items()))


def stimulus_label(row: Dict[str, str], kind: str) -> str:
    """One short name for what an arm played in one block."""
    if kind == "sequences":
        value = pick(row, "pattern") or ""
        if value.strip().lower() in SILENT:
            return "silent"
        return sequence_name(value)
    if kind == "intervals":
        value = pick(row, "interval") or ""
    else:
        value = pick(row, "frequency") or ""
    if value.strip().lower() in SILENT:
        return "silent"
    if kind == "pure tones":
        hz = as_float(value)
        if hz is not None:
            return f"{hz / 1000:g} kHz"
    return value


# -- one session --------------------------------------------------------------
@dataclass
class Session:
    mouse: str
    date: str
    start_time: str
    sex: str
    birth_date: str
    ear_mark: str
    arms: int
    kind: str
    blocks: int
    sound_blocks: int
    silent_blocks: int
    silent_block_min: str
    sound_block_min: str
    silent_arm_in_sound_blocks: str
    tonal_centre_hz: str
    stimuli: str
    total_time_s: float
    total_visits: int
    parts: int
    folders: str
    files: str
    note: str
    layout: str
    rows: List[dict] = field(default_factory=list, repr=False)


def _minutes(values: List[float]) -> str:
    """Block lengths as they occur, e.g. '2, 10' rather than one median."""
    return ", ".join(f"{v:g}" for v in sorted(set(values))) if values else ""


def read_session(folder: Path, trials: Path) -> Optional[Session]:
    rows = read_rows(trials)
    if not rows:
        return None
    columns = [c for c in rows[0] if c and not c.startswith("Unnamed")]
    kind = stimulus_kind(columns)

    m = re.search(r"mouse\s*(\d+)", folder.name, re.I)
    meta = read_mouse(folder)
    mouse = m.group(1) if m else re.sub(r"\D", "", meta["mouse_file_id"])

    starts = [as_float(pick(r, "trial_start_time")) for r in rows]
    first = min((s for s in starts if s), default=None)
    when = datetime.fromtimestamp(first) if first else None

    by_block: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_block[pick(r, "trial_ID") or "?"].append(r)

    long_rows, labels = [], set()
    silent_lengths, sound_lengths = [], []
    silent_blocks, silent_arm = 0, False
    total_ms, total_visits = 0.0, 0
    for block, block_rows in by_block.items():
        block_labels = [stimulus_label(r, kind) for r in block_rows]
        s = as_float(pick(block_rows[0], "trial_start_time"))
        e = as_float(pick(block_rows[0], "end_trial_time"))
        length = round((e - s) / 60.0, 1) if s and e and e > s else None
        is_silent = all(lab == "silent" for lab in block_labels)
        if is_silent:
            silent_blocks += 1
            if length is not None:
                silent_lengths.append(length)
        else:
            if length is not None:
                sound_lengths.append(length)
            if "silent" in block_labels:
                silent_arm = True
        for r, label in zip(block_rows, block_labels):
            if label != "silent":
                labels.add(label)
            t_ms = as_float(pick(r, "time_spent", "time spent")) or 0.0
            visits = int(as_float(pick(r, "visitation_count", "visitation count")) or 0)
            total_ms += t_ms
            total_visits += visits
            long_rows.append({
                "mouse": mouse, "date": when.strftime("%Y-%m-%d") if when else "",
                "part": 1, "folder": folder.name, "trials_file": trials.name,
                "kind": kind, "block": block,
                "block_type": "silent" if is_silent else "sound",
                "block_min": length if length is not None else "",
                "arm": pick(r, "ROIs") or "",
                "stimulus": label,
                "tones": sequence_tones(r) if kind == "sequences" else "",
                "time_s": round(t_ms / 1000.0, 3),
                "visits": visits,
            })

    # In the interval sessions every arm plays the tonal centre plus an
    # interval above it, so the lowest frequency logged is the tonal centre.
    centre = ""
    if kind == "intervals":
        freqs = [f for r in rows for f in numbers_in(pick(r, "frequencies", "frequency"))
                 if f > 0]
        if freqs:
            centre = f"{min(freqs):g}"

    return Session(
        mouse=mouse, date=when.strftime("%Y-%m-%d") if when else "",
        start_time=when.strftime("%H:%M") if when else "",
        sex=meta["sex"], birth_date=meta["birth_date"], ear_mark=meta["ear_mark"],
        arms=len({r["arm"] for r in long_rows}), kind=kind,
        blocks=len(by_block), sound_blocks=len(by_block) - silent_blocks,
        silent_blocks=silent_blocks,
        silent_block_min=_minutes(silent_lengths),
        sound_block_min=_minutes(sound_lengths),
        silent_arm_in_sound_blocks="yes" if silent_arm else "no",
        tonal_centre_hz=centre,
        stimuli=" | ".join(sorted(labels, key=_natural)),
        total_time_s=round(total_ms / 1000.0, 1), total_visits=total_visits,
        parts=1, folders=folder.name, files=trials.name, note="",
        layout=", ".join(columns), rows=long_rows,
    )


def _natural(text: str):
    """Sort 1 kHz before 10 kHz, and words alphabetically."""
    num = re.match(r"^(\d+(?:\.\d+)?)", text)
    return (0, float(num.group(1)), text) if num else (1, 0.0, text)


# -- one mouse, one day ---------------------------------------------------------
def fingerprint(s: Session) -> tuple:
    """What a file recorded, ignoring how the numbers were written.

    One folder holds the same recording saved twice, once with times written
    as 59178.0 and once as 59178. Comparing values rather than text catches
    that, so a copy is not mistaken for a restart.
    """
    return tuple(sorted((r["block"], r["arm"], r["stimulus"], r["time_s"],
                         r["visits"]) for r in s.rows))


def merge_day(recordings: List[Session]) -> Session:
    """Everything one mouse did on one day, as one session.

    Copies of the same recording collapse to one. Different recordings are
    the parts of a restarted session: their blocks are kept, numbered by part
    in the order they started, and the totals are added.
    """
    distinct: Dict[tuple, Session] = {}
    copies: Dict[tuple, List[str]] = defaultdict(list)
    for r in recordings:
        key = fingerprint(r)
        copies[key].append(f"{r.folders}/{r.files}")
        distinct.setdefault(key, r)

    parts = sorted(distinct.values(), key=lambda s: s.start_time)
    notes = []
    for key, names in copies.items():
        if len(names) > 1:
            notes.append("same recording saved " + str(len(names)) + " times: "
                         + "; ".join(names))
    if len(parts) > 1:
        notes.append(f"restarted: {len(parts)} parts merged, starting "
                     + ", ".join(p.start_time for p in parts))

    if len(parts) == 1:
        s = parts[0]
        s.files = "; ".join(sorted({n.split('/', 1)[1] for n in copies[fingerprint(s)]}))
        s.note = "; ".join(notes)
        return s

    rows = []
    for i, p in enumerate(parts, start=1):
        for r in p.rows:
            rows.append({**r, "part": i})
    first = parts[0]
    merged = Session(
        mouse=first.mouse, date=first.date, start_time=first.start_time,
        sex=first.sex, birth_date=first.birth_date, ear_mark=first.ear_mark,
        arms=max(p.arms for p in parts),
        kind=" + ".join(sorted({p.kind for p in parts})),
        blocks=sum(p.blocks for p in parts),
        sound_blocks=sum(p.sound_blocks for p in parts),
        silent_blocks=sum(p.silent_blocks for p in parts),
        silent_block_min=_minutes([float(x) for p in parts
                                   for x in p.silent_block_min.split(", ") if x]),
        sound_block_min=_minutes([float(x) for p in parts
                                  for x in p.sound_block_min.split(", ") if x]),
        silent_arm_in_sound_blocks=("yes" if any(p.silent_arm_in_sound_blocks == "yes"
                                                 for p in parts) else "no"),
        tonal_centre_hz=", ".join(sorted({p.tonal_centre_hz for p in parts
                                          if p.tonal_centre_hz})),
        stimuli=" | ".join(sorted({x for p in parts for x in p.stimuli.split(" | ") if x},
                                  key=_natural)),
        total_time_s=round(sum(p.total_time_s for p in parts), 1),
        total_visits=sum(p.total_visits for p in parts),
        parts=len(parts),
        folders="; ".join(sorted({p.folders for p in parts})),
        files="; ".join(n for names in copies.values() for n in names),
        note="; ".join(notes),
        layout=first.layout, rows=rows,
    )
    return merged


def survey(root: Path) -> List[Session]:
    recordings: Dict[tuple, List[Session]] = defaultdict(list)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if LATER.match(folder.name):
            continue
        for trials in sorted(folder.glob("trials*.csv")):
            s = read_session(folder, trials)
            if s is not None:
                recordings[(s.mouse, s.date)].append(s)
    sessions = [merge_day(rs) for rs in recordings.values()]
    sessions.sort(key=lambda s: (s.date, s.start_time, s.mouse))
    return sessions


# -- output ---------------------------------------------------------------------
def cohort_of(mouse: str) -> str:
    """Mice ordered in one batch share their leading digits (6704 to 6708)."""
    return mouse[:2] + "xx" if len(mouse) >= 3 else mouse


def describe_mouse(mouse: str, sessions: List[Session]) -> str:
    """A mouse's sex, birth date and ear mark, as most of its sessions record them.

    The per-mouse file was typed in at the start of every session, so the same
    animal can be recorded differently from one day to the next: one mouse is
    female in its first session and male in the six after. Reporting whatever
    the first session says would hide that, so this gives the value most
    sessions agree on and says when they do not.
    """
    def common(attr: str):
        values = [getattr(s, attr) for s in sessions if getattr(s, attr)]
        if not values:
            return "?", ""
        counts = Counter(values)
        top, n = counts.most_common(1)[0]
        if len(counts) == 1:
            return top, ""
        others = ", ".join(f"{v!r} x{c}" for v, c in counts.items() if v != top)
        return top, f" [{attr.replace('_', ' ')} also recorded as {others}]"

    sex, sex_note = common("sex")
    born, born_note = common("birth_date")
    ear, ear_note = common("ear_mark")
    return (f"mouse {mouse}: sex {sex}, born {born}, ear mark {ear}"
            f"{sex_note}{ear_note}")


def report(sessions: List[Session]) -> None:
    n_files = sum(len(s.files.split("; ")) for s in sessions)
    print(f"{len(sessions)} sessions (one mouse, one day) from {n_files} trials files\n")

    groups: Dict[str, List[Session]] = defaultdict(list)
    for s in sessions:
        groups[cohort_of(s.mouse)].append(s)

    for name, ss in sorted(groups.items(), key=lambda kv: min(s.date for s in kv[1])):
        mice = sorted({s.mouse for s in ss})
        print("=" * 78)
        print(f"Cohort {name}: {len(mice)} mice, {len(ss)} sessions, "
              f"{min(s.date for s in ss)} to {max(s.date for s in ss)}")
        for m in mice:
            print("   " + describe_mouse(m, [s for s in ss if s.mouse == m]))
        print()
        print(f"   {'date':<10} {'mouse':<6} {'arms':>4} {'kind':<11} "
              f"{'silent min':<10} {'sound min':<9} {'quiet arm':<9}  stimuli")
        for s in ss:
            print(f"   {s.date:<10} {s.mouse:<6} {s.arms:>4} {s.kind:<11} "
                  f"{s.silent_block_min:<10} {s.sound_block_min:<9} "
                  f"{s.silent_arm_in_sound_blocks:<9}  {s.stimuli[:52]}")
            if s.note:
                print(f"   {'':<10} note: {s.note}")
        print()


def write_tables(sessions: List[Session], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    session_fields = [f for f in asdict(sessions[0]) if f != "rows"]
    with open(out / "sessions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=session_fields)
        w.writeheader()
        for s in sessions:
            d = asdict(s)
            d.pop("rows")
            w.writerow(d)
    rows = [r for s in sessions for r in s.rows]
    with open(out / "blocks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'sessions.csv'} ({len(sessions)} rows)")
    print(f"wrote {out / 'blocks.csv'} ({len(rows)} rows)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the tables (default: beside the data)")
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        print(f"no such folder: {args.root}")
        return 1
    out = args.out or args.root.parent / (args.root.name + "_survey")

    sessions = survey(args.root)
    if not sessions:
        print("no standalone sessions found")
        return 1
    report(sessions)
    write_tables(sessions, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
