#!/usr/bin/env python3
"""
3D Views of the Complexity Figures
==================================

Figure 3.4 (the four arm pilot) and Figure 4.12 (Experiment 1) place every
stimulus by its temporal and spectral complexity. These views raise each
stimulus to a third value, so that preference can be read over both measures
at once, and draw it with plotly so that the view can be turned with the mouse.

    the pilot: the share of its time in the arms that each mouse spent on the
        arm playing the stimulus, in every kind of session (four arms, one
        stimulus to an arm, so chance is 25%); or, for the intervals alone,
        the preference index against the silent arm, as in Figure 3.4b
    Experiment 1: the preference index against the silent arm from the mean
        of its set, as in Figure 4.12b and c; or the share of the time in the
        arms (eight arms, chance 12.5%)

A menu above each view switches between the two. The stimuli lie on the floor
as in panel (a) of each figure, and a stem runs from chance (or from an index
of zero) up or down to the mean over mice, with its standard error; for the
index of Experiment 1, to the median with its quartiles, as in Figure 4.12.
The shares of one block add up to 100%, so their mean is the one to set
against chance; a few mice that stay long on one arm pull the median below it. Each mouse can be shown from the legend. Stimuli that share
a place in complexity space are set slightly apart around it; the hover text
gives their true values. The pure tones of the pilot all sit at one place and
are shown one by one, since pooled they fill every arm and take 25% by
construction.

Usage:
    python complexity_3d.py [--survey DIR] [--exp1 DIR] [--gallery DIR]

Reads the pilot's blocks.csv through pilot_four_arm.py, and Experiment 1's
complexity/complexity.csv (exp1_complexity.py), stimulus_pi/stimulus_pi.csv
(exp1_stimulus_pi.py) and lmm/arm_blocks.csv (exp1_lmm.py). Writes
pilot_complexity_3d.html beside the pilot's figures (SURVEY/pilot/) and
exp1_complexity_3d.html beside Experiment 1's (EXP1/BATCH_ANALYSIS/complexity/),
each a page that opens without a connection, and the numbers the thesis quotes
from these views to complexity/stats_3d.txt: the shares by kind of stimulus, the
shares of the silent and vocalisation arms, and the complex tones of the envelope
day against the smooth and rough ones (Wilcoxon signed rank over mice). With
--gallery, each view is also written there as JSON, for a page that loads plotly
itself.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_four_arm as pilot
from exp1_stimulus_pi import COLOURS, label

FONT = "Georgia, 'Times New Roman', serif"
GREY = "#9E9E9E"
PILOT_SYMBOL = {"tone": "circle-open", "unison": "circle", "consonant": "circle",
                "dissonant": "circle", "sequence": "diamond"}
PILOT_FAMILIES = [("tone", "pure tones"), ("unison", "unison"), ("consonant", "consonant"),
                  ("dissonant", "dissonant"), ("sequence", "sequences")]
SET_SYMBOL = {"TEM": "circle", "Intervals 1": "square", "Intervals 2": "diamond",
              "Sequences": "cross"}


# -- the data -------------------------------------------------------------------
def shares(rows, key, block_key):
    """Each mouse's share of its time in the arms on each stimulus, summed over the
    blocks in which the stimulus played."""
    rows = rows.copy()
    rows["block_time"] = rows.groupby(block_key)["time_s"].transform("sum")
    rows = rows[rows["block_time"] > 0]
    g = rows.groupby(key, as_index=False)[["time_s", "block_time"]].sum()
    g["value"] = 100 * g["time_s"] / g["block_time"]
    return g


def pilot_data(survey):
    b = pilot.load(survey)
    table = pilot.complexity_table(b)
    s = b[b["block_type"] == "sound"]
    share = shares(s, ["mouse", "kind", "stimulus"], ["mouse", "date", "part", "block"])
    share = share[share["stimulus"] != "silent"]
    idx = pilot.stimulus_indices(b)
    idx = idx[idx["stimulus"] != "all sounds"].assign(kind="intervals", value=lambda x: x["pi"])
    places = table.assign(key=table["kind"] + "|" + table["stimulus"])
    places["name"] = [pilot.PRETTY.get(s, f"{float(s.split()[0]):.3g} kHz" if k == "pure tones" else s)
                      for s, k in zip(places["stimulus"], places["kind"])]
    places["group"] = places["family"]
    places["colour"] = places["family"].map(pilot.FAMILY_COLOUR)
    places["symbol"] = places["family"].map(PILOT_SYMBOL)
    places["detail"] = places["kind"]
    out = {}
    for name, d in (("share", share), ("pi", idx)):
        d = d.assign(key=d["kind"] + "|" + d["stimulus"])
        out[name] = d[["key", "mouse", "value"]]
    return places, out


def exp1_data(exp1):
    base = Path(exp1) / "BATCH_ANALYSIS"
    comp = pd.read_csv(base / "complexity" / "complexity.csv")
    st = pd.read_csv(base / "stimulus_pi" / "stimulus_pi.csv")
    st = st[(st["family"] != "vocalisation") & st["set"].isin(SET_SYMBOL)].dropna(subset=["pi"])
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    # as in Figure 4.12, each index is taken from the mean of its set
    st["value"] = st["pi"] - st.groupby("set")["pi"].transform("mean")
    arms = pd.read_csv(base / "lmm" / "arm_blocks.csv")
    share = shares(arms, ["mouse", "set", "stimulus"], ["session", "block"])
    places = comp.assign(key=comp["set"] + "|" + comp["stimulus"])
    places["name"] = places["stimulus"].map(label)
    places["group"] = places["set"]
    places["colour"] = places["family"].map(lambda f: COLOURS.get(f, "#424242"))
    places["symbol"] = places["set"].map(SET_SYMBOL)
    places["detail"] = places["set"]
    out = {}
    for name, d in (("pi", st), ("share", share)):
        d = d.assign(key=d["set"] + "|" + d["stimulus"])
        out[name] = d[d["key"].isin(places["key"])][["key", "mouse", "value"]]
    return places, out


def spread(places):
    """Set stimuli that share a place in complexity space slightly apart around it."""
    p = places.copy()
    p["x"], p["y"] = p["temporal_bits"], p["spectral"]
    rx = 0.035 * (p["x"].max() - p["x"].min())
    ry = 0.035 * (p["y"].max() - p["y"].min())
    at = p.groupby([p["x"].round(2), p["y"].round(3)]).groups
    for _, rows in at.items():
        rows = sorted(rows, key=lambda i: (p.at[i, "group"], p.at[i, "name"]))
        if len(rows) < 2:
            continue
        for k, i in enumerate(rows):
            angle = 2 * np.pi * k / len(rows) + np.pi / 2
            p.at[i, "x"] += rx * np.cos(angle)
            p.at[i, "y"] += ry * np.sin(angle)
    return p


def summary(places, mice, centre):
    rows = []
    for key, g in mice.groupby("key"):
        v = g["value"].to_numpy(float)
        if centre == "mean":
            c = v.mean()
            e = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            lo, hi = c - e, c + e
        else:
            c, lo, hi = np.median(v), np.quantile(v, 0.25), np.quantile(v, 0.75)
        rows.append({"key": key, "centre": c, "lo": lo, "hi": hi, "n": len(v)})
    return places.merge(pd.DataFrame(rows), on="key")


# -- the figure -------------------------------------------------------------------
def segments(xs, ys, z0, z1):
    """Vertical segments as one line trace, broken by gaps."""
    X, Y, Z = [], [], []
    for x, y, a, b in zip(xs, ys, z0, z1):
        X += [x, x, None]
        Y += [y, y, None]
        Z += [a, b, None]
    return X, Y, Z


def measure_traces(places, mice, m, groups, visible):
    """Every trace for one measure: per group the floor, the stems, the spread over
    mice and the centres; then the mice, the names and the chance plane."""
    s = summary(places, mice, m["centre"])
    traces = []
    floor = m["range"][0]
    for group, group_name in groups:
        g = s[s["group"] == group]
        if g.empty:
            continue
        # one colour per vertex, three to a segment, so a group can mix kinds
        colours = [c for c in g["colour"] for _ in range(3)]
        traces.append(go.Scatter3d(
            x=g["x"], y=g["y"], z=[floor] * len(g), mode="markers", legendgroup=group,
            showlegend=False, hoverinfo="skip", visible=visible,
            marker=dict(size=3.5, color=g["colour"], symbol=g["symbol"], opacity=0.35)))
        X, Y, Z = segments(g["x"], g["y"], [m["baseline"]] * len(g), g["centre"])
        traces.append(go.Scatter3d(
            x=X, y=Y, z=Z, mode="lines", legendgroup=group, showlegend=False,
            hoverinfo="skip", visible=visible, line=dict(color=colours, width=3)))
        X, Y, Z = segments(g["x"], g["y"], g["lo"], g["hi"])
        traces.append(go.Scatter3d(
            x=X, y=Y, z=Z, mode="lines", legendgroup=group, showlegend=False,
            hoverinfo="skip", visible=visible, opacity=0.3, line=dict(color=colours, width=6)))
        # a group of mixed colours (a day of Experiment 1) gets a grey marker in
        # the legend, which shows only its shape
        mixed = g["colour"].nunique() > 1
        if mixed:
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers", name=group_name,
                legendgroup=group, visible=visible, hoverinfo="skip",
                marker=dict(size=5.5, color="#616161", symbol=g["symbol"].iloc[0])))
        traces.append(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["centre"], mode="markers", name=group_name,
            legendgroup=group, visible=visible, showlegend=not mixed,
            marker=dict(size=5.5, color=g["colour"], symbol=g["symbol"],
                        line=dict(color="white", width=0.5)),
            customdata=np.stack([g["name"], g["detail"], g["temporal_bits"], g["spectral"],
                                 g["lo"], g["hi"], g["n"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                           "temporal %{customdata[2]:.2f} bits, spectral %{customdata[3]:.3f}<br>"
                           + m["hover"] + "<extra></extra>")))
    d = mice.merge(s[["key", "x", "y", "name", "colour", "symbol", "detail"]], on="key")
    traces.append(go.Scatter3d(
        x=d["x"], y=d["y"], z=d["value"], mode="markers", name="each mouse",
        visible="legendonly" if visible else False,
        marker=dict(size=2.5, color=d["colour"], opacity=0.6),
        customdata=np.stack([d["mouse"].astype(str), d["name"], d["detail"]], axis=-1),
        hovertemplate=("mouse %{customdata[0]}<br>%{customdata[1]} (%{customdata[2]})<br>"
                       + m["mouse_hover"] + "<extra></extra>")))
    traces.append(go.Scatter3d(
        x=s["x"], y=s["y"], z=s["centre"] + m["lift"], mode="text", text=s["name"],
        name="stimulus names", hoverinfo="skip",
        visible=(True if m["names"] else "legendonly") if visible else False,
        textfont=dict(size=10, color="#37474F", family=FONT)))
    x0, x1 = places["temporal_bits"].min() - 0.8, places["temporal_bits"].max() + 0.6
    y0, y1 = places["spectral"].min() - 0.01, places["spectral"].max() + 0.01
    traces.append(go.Surface(
        x=[x0, x1], y=[y0, y1], z=[[m["baseline"]] * 2] * 2, showscale=False, opacity=0.15,
        colorscale=[[0, GREY], [1, GREY]], hoverinfo="skip", name=m["plane"],
        showlegend=True, visible=visible))
    return traces


def figure(places, mice, measures, groups, title):
    places = spread(places)
    traces, owner = [], []
    for k, m in enumerate(measures):
        t = measure_traces(places, mice[m["key"]], m, groups, visible=(k == 0))
        traces += t
        owner += [k] * len(t)
    fig = go.Figure(traces)
    fig._owner, fig._measures = owner, measures

    def visibility(k):
        out = []
        for i, tr in enumerate(traces):
            if owner[i] != k:
                out.append(False)
            elif tr.name == "each mouse":
                out.append("legendonly")
            elif tr.name == "stimulus names":
                out.append(True if measures[k]["names"] else "legendonly")
            else:
                out.append(True)
        return out

    buttons = [dict(label=m["button"], method="update",
                    args=[{"visible": visibility(k)},
                          {"scene.zaxis.title.text": m["axis"], "scene.zaxis.range": m["range"]}])
               for k, m in enumerate(measures)]
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        font=dict(family=FONT, size=12, color="#212121"),
        paper_bgcolor="white", height=640, margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.92, bgcolor="rgba(255,255,255,0.8)", font=dict(size=11),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        updatemenus=[dict(type="buttons", direction="right", x=0.99, xanchor="right", y=1.06,
                          yanchor="bottom", buttons=buttons, showactive=True,
                          bgcolor="white", bordercolor="#BDBDBD", font=dict(size=11))],
        scene=dict(
            xaxis=dict(title=dict(text="Temporal complexity (bits)"), backgroundcolor="white",
                       gridcolor="#E0E0E0", zeroline=False),
            yaxis=dict(title=dict(text="Spectral complexity"), backgroundcolor="white",
                       gridcolor="#E0E0E0", zeroline=False),
            zaxis=dict(title=dict(text=measures[0]["axis"]), range=measures[0]["range"],
                       backgroundcolor="white", gridcolor="#E0E0E0", zeroline=False),
            aspectmode="manual", aspectratio=dict(x=1.6, y=1.0, z=0.9),
            camera=dict(eye=dict(x=1.35, y=-1.45, z=0.6), center=dict(x=0, y=0, z=-0.1))))
    return fig


def write_static(fig, key, path):
    """One view of a figure as a still image for the thesis, without the menu, the
    title or the mice: PNG and PDF through kaleido (which drives Chrome)."""
    k = [m["key"] for m in fig._measures].index(key)
    m = fig._measures[k]
    still = go.Figure(fig)
    for tr, who in zip(still.data, fig._owner):
        if who != k or tr.name == "each mouse":
            tr.visible = False
        elif tr.name == "stimulus names":
            tr.visible = m["names"]
            tr.showlegend = False
            # the pure tones share a place: one name for them all
            text = list(tr.text)
            tones = [i for i, t in enumerate(text) if str(t).endswith("kHz")]
            for i in tones:
                text[i] = ""
            if tones:
                text[tones[0]] = "pure tones"
            tr.text = text
        else:
            tr.visible = True
    still.layout.updatemenus = ()
    still.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0),
                        font=dict(size=17),
                        legend=dict(font=dict(size=16), x=0.99, xanchor="right", y=0.97),
                        scene=dict(zaxis=dict(title=dict(text=m["axis"]), range=m["still_range"]),
                                   camera=dict(eye=dict(x=1.3, y=-1.42, z=0.55),
                                               center=dict(x=0, y=0, z=-0.1))))
    still.update_traces(selector=dict(name="stimulus names"), textfont=dict(size=13))
    for ext in ("png", "pdf"):
        still.write_image(f"{path}.{ext}", width=1500, height=950, scale=2 if ext == "png" else 1)
    print(f"  {path}.png / .pdf")


CONFIG = {"displaylogo": False, "responsive": True,
          "toImageButtonOptions": {"format": "png", "scale": 4}}


def pilot_figure(survey):
    places, mice = pilot_data(survey)
    top = max(mice["share"]["value"].max(), 40)
    measures = [
        dict(key="share", centre="mean", baseline=25.0, range=[0, np.ceil(top / 5) * 5],
             axis="Time on the arm (% of time in the arms)", button="time on the arm",
             plane="chance, 25%", names=True, lift=1.5, still_range=[0, 40],
             hover="%{z:.1f}% of the time in the arms (SE %{customdata[4]:.1f} to "
                   "%{customdata[5]:.1f}), %{customdata[6]} mice",
             mouse_hover="%{z:.1f}% of the time in the arms"),
        dict(key="pi", centre="mean", baseline=0.0, range=[-0.6, 0.6],
             axis="Preference index against the silent arm", button="index against silence",
             plane="no preference", names=True, lift=0.04, still_range=[-0.6, 0.6],
             hover="index %{z:.2f} (SE %{customdata[4]:.2f} to %{customdata[5]:.2f}), "
                   "%{customdata[6]} mice",
             mouse_hover="index %{z:.2f}"),
    ]
    return figure(places, mice, measures, PILOT_FAMILIES,
                  "Figure 3.4 in three dimensions, the four arm pilot"), places, mice


def exp1_figure(exp1):
    places, mice = exp1_data(exp1)
    measures = [
        dict(key="pi", centre="median", baseline=0.0, range=[-0.25, 0.25],
             axis="Index against silence, from its set's mean", button="index against silence",
             plane="the set's mean", names=False, lift=0.02, still_range=[-0.25, 0.25],
             hover="median %{z:.3f} (quartiles %{customdata[4]:.3f} to %{customdata[5]:.3f}), "
                   "%{customdata[6]} mice",
             mouse_hover="index from the set's mean %{z:.3f}"),
        # the shares of one block add up to 100%, so their mean, not their
        # median, is the one to set against chance
        dict(key="share", centre="mean", baseline=12.5, range=[0, 30],
             axis="Time on the arm (% of time in the arms)", button="time on the arm",
             plane="chance, 12.5%", names=False, lift=0.8, still_range=[0, 20],
             hover="mean %{z:.1f}% of the time in the arms (SE %{customdata[4]:.1f} "
                   "to %{customdata[5]:.1f}), %{customdata[6]} mice",
             mouse_hover="%{z:.1f}% of the time in the arms"),
    ]
    return figure(places, mice, measures, [(s, s) for s in SET_SYMBOL],
                  "Figure 4.12 in three dimensions, Experiment 1"), places, mice


def share_stats(exp1, pilot_places, pilot_mice, say):
    """The numbers the thesis quotes from these views: the pilot's shares by kind
    of stimulus; in Experiment 1, the shares of the silent and the vocalisation
    arm on each day, and the complex tones of the envelope day against the smooth
    and rough ones, in time, entries and visit length (Wilcoxon signed rank over
    mice, each mouse's arms of a kind pooled)."""
    from scipy.stats import wilcoxon
    s = summary(pilot_places, pilot_mice["share"], "mean")
    say("The pilot, mean share of the time in the arms (%) over mice, chance 25:")
    for fam, g in s.groupby("family"):
        say(f"  {fam:10s} {g['centre'].min():5.1f} to {g['centre'].max():5.1f}  "
            + ", ".join(f"{n} {c:.1f}" for n, c in zip(g["name"], g["centre"])))

    arms = pd.read_csv(Path(exp1) / "BATCH_ANALYSIS" / "lmm" / "arm_blocks.csv")
    arms = arms[arms["set"].isin(SET_SYMBOL)].copy()
    arms["block_time"] = arms.groupby(["session", "block"])["time_s"].transform("sum")
    # a block with no time in the arms adds nothing to a pooled share, and counts
    # as no entries; every row is one arm in one block and carries the block's whole time, so the
    # pooled share is that of one arm of the class
    per = arms.groupby(["mouse", "set", "cls"])[["time_s", "block_time"]].sum()
    per["share"] = 100 * per["time_s"] / per["block_time"]
    share = per["share"].unstack("cls")
    say("\nExperiment 1, mean share of the time in the arms (%) for one arm of each class, "
        "chance 12.5:")
    say(share.groupby("set").mean().round(1).to_string())

    t = arms[(arms["set"] == "TEM") & (arms["cls"] == "synthetic")].copy()
    t["kind"] = np.where(t["family"] == "rough_complex", "complex", "smooth or rough")
    g = t.groupby(["mouse", "kind"]).agg(time=("time_s", "sum"), block_time=("block_time", "sum"),
                                         visits=("visits", "sum"), arm_blocks=("arm", "size"))
    g["share"] = 100 * g["time"] / g["block_time"]
    g["entries"] = g["visits"] / g["arm_blocks"]
    g["visit_s"] = g["time"] / g["visits"]
    say("\nTemporal envelope modulation, the complex tones (one arm each) against the smooth "
        "and rough ones, per mouse:")
    for col, name in (("share", "share of the time in the arms (%)"),
                      ("entries", "entries per arm in a block"), ("visit_s", "mean visit (s)")):
        w = g[col].unstack().dropna()
        p = wilcoxon(w["complex"], w["smooth or rough"]).pvalue
        say(f"  {name:36s} complex {w['complex'].mean():.2f}, smooth or rough "
            f"{w['smooth or rough'].mean():.2f}, higher in {int((w['complex'] > w['smooth or rough']).sum())} "
            f"of {len(w)} mice, p = {p:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--survey", type=Path, default=pilot.SURVEY)
    parser.add_argument("--exp1", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="also write each view as JSON into this folder")
    parser.add_argument("--thesis-figures", type=Path, default=None,
                        help="also write the time on the arm view of each as a still "
                             "image, fig_pilot_complexity_3d and fig_exp1_complexity_3d "
                             "(needs kaleido)")
    args = parser.parse_args()

    fig, pilot_places, pilot_mice = pilot_figure(args.survey)
    path = args.survey / "pilot" / "pilot_complexity_3d.html"
    fig.write_html(path, include_plotlyjs=True, config=CONFIG)
    print(f"  {path}")
    fig1, places, mice = exp1_figure(args.exp1)
    path1 = Path(args.exp1) / "BATCH_ANALYSIS" / "complexity" / "exp1_complexity_3d.html"
    fig1.write_html(path1, include_plotlyjs=True, config=CONFIG)
    print(f"  {path1}")
    if args.gallery:
        args.gallery.mkdir(parents=True, exist_ok=True)
        (args.gallery / "pilot_complexity_3d.json").write_text(fig.to_json(), encoding="utf-8")
        (args.gallery / "exp1_complexity_3d.json").write_text(fig1.to_json(), encoding="utf-8")
        print(f"  JSON in {args.gallery}")
    if args.thesis_figures:
        args.thesis_figures.mkdir(parents=True, exist_ok=True)
        write_static(fig, "share", str(args.thesis_figures / "fig_pilot_complexity_3d"))
        write_static(fig1, "share", str(args.thesis_figures / "fig_exp1_complexity_3d"))

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    share_stats(args.exp1, pilot_places, pilot_mice, say)
    s = summary(places, mice["share"], "mean")
    say("\nExperiment 1, mean share of the time in the arms (%) for each stimulus, chance 12.5:")
    say(s.sort_values(["group", "temporal_bits"])[["group", "name", "centre", "n"]]
        .round(2).to_string(index=False))
    stats = Path(args.exp1) / "BATCH_ANALYSIS" / "complexity" / "stats_3d.txt"
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  {stats}")


if __name__ == "__main__":
    main()
