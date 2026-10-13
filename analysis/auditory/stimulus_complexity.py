#!/usr/bin/env python3
"""Two measures of how complex a stimulus is, beyond sound against silence.

Spectral complexity: how widely the sound's energy is spread across frequency.
    The spectral entropy of its power spectrum over everything the sound card
    can produce, 0 to 96 kHz, normalised so that 0 is all the energy in one
    frequency bin and 1 is energy spread evenly over every bin. A pure tone is
    low, about 0.10 whatever its frequency (the floor set by the analysis
    window); two tones sit higher; a vocalisation or noise much higher. The
    band is the whole spectrum rather than the range mice hear because a band
    edge cuts a tone near it in half: a 1 kHz tone measured from 1 kHz reads
    0.03 instead of 0.10. It says nothing about order in time: ABAB and a
    random sequence of the same tones have nearly the same spectrum.

Temporal complexity: how long the sound runs before it repeats, counted in
    its own elementary units and given in bits (log2 of that count). It is
    how much of the sound must be held in memory to predict the rest of it
    exactly.

        pure tone             repeats every cycle                  0 bits
        interval p/q          repeats every q cycles of the        log2 q
                              lower tone (relative periodicity,
                              Stolzenburg 2015)
        sequence              repeats every pattern length,        log2 of the
                              counted in slots                     period
        random sequence       does not repeat within the stimulus  log2 of its
                                                                   length

    For intervals this is the periodicity account of consonance: the simple
    ratios that sound consonant repeat quickly (a fifth, 3/2, every 2 cycles)
    and the dissonant ones slowly (a tritone, 64/45, every 45).

The spectral measure is computed from the waveform, synthesised here with
equal amplitudes for every tone, which is the sound at the animal's ear if the
speaker correction did its job. The temporal measure is exact, from the
stimulus's definition.

    python analysis/auditory/stimulus_complexity.py     prints a worked table
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import welch

FS = 192000
BAND_HZ = (0.0, FS / 2)
RAMP_S = 0.004

# The intervals as they were played (just intonation). The tritone was played
# as 64/45 although the code labelled it 45/32.
RATIOS: Dict[str, Fraction] = {
    "unison": Fraction(1, 1), "min_2": Fraction(16, 15), "maj_2": Fraction(9, 8),
    "min_3": Fraction(6, 5), "maj_3": Fraction(5, 4), "perf_4": Fraction(4, 3),
    "tritone": Fraction(64, 45), "perf_5": Fraction(3, 2), "min_6": Fraction(8, 5),
    "maj_6": Fraction(5, 3), "min_7": Fraction(16, 9), "maj_7": Fraction(15, 8),
    "octave": Fraction(2, 1),
}
CONSONANT = {"unison", "min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"}
DISSONANT = {"min_2", "maj_2", "tritone", "min_7", "maj_7"}


# -- synthesis ----------------------------------------------------------------
def _ramped(n: int) -> np.ndarray:
    env = np.ones(n)
    r = min(int(RAMP_S * FS), n // 2)
    if r:
        ramp = np.sin(0.5 * np.pi * np.linspace(0, 1, r)) ** 2
        env[:r] = ramp
        env[-r:] = ramp[::-1]
    return env


def tones(freqs_hz: Sequence[float], duration_s: float = 1.0) -> np.ndarray:
    """Equal-amplitude sines sounding together, with onset and offset ramps."""
    t = np.arange(int(duration_s * FS)) / FS
    w = sum(np.sin(2 * np.pi * f * t) for f in freqs_hz if f > 0)
    if np.isscalar(w):
        return np.zeros_like(t)
    return w * _ramped(len(t)) / max(1, len([f for f in freqs_hz if f > 0]))


def sequence_wave(symbols: str, tone_of: Dict[str, float], slot_s: float = 0.04) -> np.ndarray:
    """A sequence of slots, each a ramped tone, or silence for 'o'."""
    pieces = [tones([tone_of[s]], slot_s) if s in tone_of else np.zeros(int(slot_s * FS))
              for s in symbols]
    return np.concatenate(pieces)


# -- spectral complexity ------------------------------------------------------------
def spectral_entropy(wave: np.ndarray, band=BAND_HZ) -> float:
    """Normalised spectral entropy, 0 (one frequency) to 1 (flat), over band."""
    if not np.any(wave):
        return float("nan")
    f, p = welch(wave, fs=FS, nperseg=8192)
    keep = (f >= band[0]) & (f <= band[1])
    p = p[keep]
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / np.log2(len(p)))


TEMPORAL_FRAME_S = 0.005


def temporal_entropy(wave: np.ndarray, low_hz: float = 0.0,
                     frame_s: float = TEMPORAL_FRAME_S) -> float:
    """Normalised temporal entropy, 0 (all the energy in one frame) to 1 (steady).

    The counterpart in time of spectral_entropy: how evenly a sound's energy
    is spread over its duration, after Sueur et al. (2008), on energy rather
    than amplitude, as the spectral measure is. The sound is high-passed at
    low_hz, cut into frames of frame_s (5 ms resolves a call of 10 to 100 ms),
    and the share of the total energy in each frame taken as a distribution.
    A single call in silence is low, continuous calling or steady noise high.
    It is meant for recordings, which never repeat. The synthetic stimuli of
    Experiment 1 are steady or regular at this grain and sit between 0.89
    (AoAo, half its slots silent) and 1 (a pure tone).
    """
    if not np.any(wave):
        return float("nan")
    x = np.asarray(wave, dtype=float)
    if low_hz > 0:
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(8, low_hz, btype="highpass", fs=FS, output="sos"), x)
    n = int(round(frame_s * FS))
    k = len(x) // n
    if k < 2:
        return float("nan")
    e = (x[:k * n] ** 2).reshape(k, n).sum(axis=1)
    total = e.sum()
    if total <= 0:
        return float("nan")
    p = e / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / np.log2(k))


# -- temporal complexity --------------------------------------------------------------
def interval_periodicity_bits(name: str) -> float:
    """log2 of the cycles of the lower tone before a dyad repeats."""
    return math.log2(RATIOS[name].denominator)


def pattern_period(symbols: str) -> Optional[int]:
    """The shortest unit a symbol string repeats, or None if it never does."""
    n = len(symbols)
    for k in range(1, n // 2 + 1):
        if all(symbols[i] == symbols[i % k] for i in range(n)):
            return k
    return None


def sequence_periodicity_bits(pattern: str, length_slots: int = 200) -> float:
    """log2 of the slots before a sequence repeats; its length if it never does."""
    if pattern == "random":
        return math.log2(length_slots)
    period = pattern_period((pattern * (length_slots // len(pattern) + 1))[:length_slots])
    return math.log2(period if period else length_slots)


# -- one row per stimulus -------------------------------------------------------------
def describe(kind: str, label: str, *, centre_hz: float = 10000.0,
             tone_hz: Optional[float] = None, seq_tones: Optional[Dict[str, float]] = None,
             seq_length: int = 200, seed: int = 0) -> Dict[str, object]:
    """Both complexities for one stimulus, plus what kind of stimulus it is."""
    if label == "silent":
        return {"kind": kind, "stimulus": label, "spectral": float("nan"),
                "temporal_bits": float("nan"), "family": "silence"}
    if kind == "intervals":
        ratio = RATIOS[label]
        wave = tones([centre_hz, centre_hz * float(ratio)])
        family = ("unison" if label == "unison" else
                  "consonant" if label in CONSONANT else "dissonant")
        return {"kind": kind, "stimulus": label, "ratio": str(ratio),
                "spectral": spectral_entropy(wave),
                "temporal_bits": interval_periodicity_bits(label), "family": family}
    if kind == "pure tones":
        f = tone_hz if tone_hz is not None else float(label.split()[0]) * 1000
        return {"kind": kind, "stimulus": label, "spectral": spectral_entropy(tones([f])),
                "temporal_bits": 0.0, "family": "tone"}
    if kind == "sequences":
        tone_of = seq_tones or {"A": 10000.0, "B": 14000.0, "C": 16000.0}
        if label == "random":
            rng = np.random.default_rng(seed)
            symbols = "".join(rng.choice(list("ABCo"), size=seq_length))
        else:
            unit = label[: len(label) // 2] if len(label) % 2 == 0 else label
            symbols = (unit * (seq_length // len(unit) + 1))[:seq_length]
        return {"kind": kind, "stimulus": label,
                "spectral": spectral_entropy(sequence_wave(symbols, tone_of)),
                "temporal_bits": sequence_periodicity_bits(label, seq_length),
                "family": "sequence"}
    raise ValueError(f"unknown kind {kind!r}")


# -- one temporal scale for every periodic stimulus ---------------------------
# The measures above count in each family's own unit: cycles of the lower tone
# for an interval, slots for a sequence. To set stimuli of different families
# on one axis, temporal complexity is counted in one unit for all of them: the
# cycles of the sound's lowest frequency in one repetition of the waveform,
#
#     bits = log2(period x lowest frequency)
#
# For two tones at p/q this is log2 q, the measure above. A pure tone repeats
# every cycle (0 bits); a carrier amplitude-modulated at m Hz repeats every
# 1/m s; a sequence repeats with its pattern of slots. A recording that never
# repeats has no value on this scale.

def cycles_bits(period_s: float, lowest_hz: float) -> float:
    """log2 of the cycles of the lowest frequency in one period."""
    return math.log2(max(1.0, round(period_s * lowest_hz)))


def am_wave(carrier_hz: float, rates_hz: Sequence[float], depth: float = 0.5,
            segment_s: float = 0.2, duration_s: float = 1.2) -> np.ndarray:
    """A carrier whose envelope is modulated at each rate in turn, as Experiment 1 did.

    One rate gives a steady modulation; several are cycled through in
    segments of segment_s, with the envelope's phase restarting in each.
    """
    t = np.arange(int(duration_s * FS)) / FS
    seg = int(segment_s * FS)
    env = np.empty_like(t)
    for k in range(int(np.ceil(len(t) / seg))):
        rate = rates_hz[k % len(rates_hz)]
        local = t[k * seg:(k + 1) * seg] - t[k * seg]
        env[k * seg:(k + 1) * seg] = 1 + depth * np.sin(2 * np.pi * rate * local)
    return np.sin(2 * np.pi * carrier_hz * t) * env * _ramped(len(t))


EXP1_INTERVAL_BASE_HZ = 15000.0     # every interval and unison of Experiment 1
EXP1_SEQUENCE_TONES = {"A": 10000.0, "B": 15000.0, "C": 20000.0}
EXP1_SLOT_S = 0.04
EXP1_TEM_RATES = {"smooth": [], "rough": [50.0], "rough_complex": [30.0, 50.0, 70.0]}


def describe_exp1(name: str, family: str) -> Dict[str, object]:
    """Both complexities of an Experiment 1 stimulus, on the common temporal scale.

    name and family as exp1_stimulus_pi.py labels them: 'rough 10 kHz' with
    family 'rough', 'perf_5' with family 'consonant', 'AM unison', 'ABAB', and
    so on. Vocalisations return NaN.
    """
    if family in ("vocalisation", "silent"):
        return {"stimulus": name, "family": family, "spectral": float("nan"),
                "temporal_bits": float("nan")}
    if family in EXP1_TEM_RATES:                             # temporal envelope modulation
        carrier = float(name.split()[-2]) * 1000
        rates = EXP1_TEM_RATES[family]
        if not rates:
            wave, bits = tones([carrier]), 0.0
        else:
            wave = am_wave(carrier, rates)
            period = len(rates) * 0.2 if len(rates) > 1 else 1 / rates[0]
            bits = cycles_bits(period, carrier - max(rates))
        return {"stimulus": name, "family": family, "spectral": spectral_entropy(wave),
                "temporal_bits": bits}
    if family == "unison":
        base = EXP1_INTERVAL_BASE_HZ
        if name == "AM unison":
            return {"stimulus": name, "family": family,
                    "spectral": spectral_entropy(am_wave(base, [50.0])),
                    "temporal_bits": cycles_bits(1 / 50.0, base - 50.0)}
        return {"stimulus": name, "family": family, "spectral": spectral_entropy(tones([base])),
                "temporal_bits": 0.0}
    if family in ("consonant", "dissonant"):
        base = EXP1_INTERVAL_BASE_HZ
        return {"stimulus": name, "family": family,
                "spectral": spectral_entropy(tones([base, base * float(RATIOS[name])])),
                "temporal_bits": interval_periodicity_bits(name)}
    if family == "sequence":
        # the shortest unit the pattern repeats: 'A' for AAAAA, 'AB' for ABAB,
        # but the whole of ABBA
        unit = name[: pattern_period(name * 4) or len(name)]
        symbols = (unit * (200 // len(unit) + 1))[:200]
        lowest = min(EXP1_SEQUENCE_TONES[c] for c in set(unit) if c in EXP1_SEQUENCE_TONES)
        return {"stimulus": name, "family": family,
                "spectral": spectral_entropy(sequence_wave(symbols, EXP1_SEQUENCE_TONES,
                                                           EXP1_SLOT_S)),
                "temporal_bits": cycles_bits(len(unit) * EXP1_SLOT_S, lowest)}
    raise ValueError(f"unknown family {family!r} for {name!r}")


if __name__ == "__main__":
    rows = [describe("pure tones", "10 kHz")]
    rows += [describe("intervals", n) for n in
             ("unison", "perf_5", "perf_4", "maj_6", "maj_3", "maj_2", "maj_7",
              "min_2", "tritone")]
    rows += [describe("sequences", p) for p in ("AAAA", "AoAo", "ABAB", "ABCABC", "random")]
    print(f"{'kind':<11} {'stimulus':<9} {'ratio':<6} {'spectral':>9} {'temporal (bits)':>16}")
    for r in rows:
        print(f"{r['kind']:<11} {r['stimulus']:<9} {r.get('ratio', ''):<6} "
              f"{r['spectral']:>9.3f} {r['temporal_bits']:>16.2f}")
