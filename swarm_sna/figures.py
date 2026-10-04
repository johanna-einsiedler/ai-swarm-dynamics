"""Figures. One house style, shared by every command that draws.

Palette: the three largest model families get a validated categorical hue and
everything else is neutral grey. Eight families on one scatter cannot be told
apart by a colourblind reader, and a family league table is not the point.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, FAINT = "#1f2328", "#6b7280", "#d7dae0"
NULL_FILL, OBSERVED = "#c9ced6", "#c2410c"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated for all pairs; a 4th slot would not be
OTHER = "#9aa0a8"
NAMED_FAMILIES = ["Anthropic", "OpenAI", "Google"]


def family_colour(fam):
    return SERIES[NAMED_FAMILIES.index(fam)] if fam in NAMED_FAMILIES else OTHER


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax


def _legend(ax, labels, colours, **kw):
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c) for c in colours]
    leg = ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, **kw)
    return leg


def networks(W, names, families, groups, path, top_edges=120):
    """One panel per group. Node size = in-strength, colour = family, only the strongest edges drawn.

    `names` is the node order of W; `families` maps a name to its model family.
    """
    import networkx as nx

    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.8), squeeze=False)
    for ax, M, g in zip(axes[0], W, groups):
        ax.set_axis_off()
        keep = (M.sum(0) + M.sum(1)) > 0
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            continue
        sub = M[np.ix_(idx, idx)]
        G = nx.from_numpy_array(sub, create_using=nx.DiGraph)
        pos = nx.spring_layout(G, seed=0, weight="weight", k=1.6 / np.sqrt(len(idx)))
        edges = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])[:top_edges]
        wmax = max((e[2]["weight"] for e in edges), default=1)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v) for u, v, _ in edges], ax=ax, edge_color=FAINT,
            width=[0.3 + 2.2 * (d["weight"] / wmax) for _, _, d in edges], arrows=False,
        )
        ins = sub.sum(0)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=60 + 900 * ins / (ins.max() or 1),
            node_color=[family_colour(families.get(names[i])) for i in idx], linewidths=0.8, edgecolors="white",
        )
        for rank in np.argsort(-ins)[:4]:
            ax.annotate(names[idx[rank]], pos[rank], fontsize=7.5, color=INK, ha="center", va="center", xytext=(0, 13), textcoords="offset points")
        ax.set_title(f"{g}  ({len(idx)} agents)", fontsize=10, color=INK, loc="left")
    _legend(axes[0][0], NAMED_FAMILIES + ["other"], SERIES + [OTHER], loc="lower left", ncol=2)
    fig.suptitle("Mention network: node size is mentions received, strongest ties only", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ladder(table, path):
    """Completeness per rung, with its bootstrap interval."""
    t = table[table.model != "no covariates"]
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.52 * len(t) + 1.4))
    ax.axvline(1.0, color=FAINT, linewidth=1, zorder=0)
    ax.hlines(y, t.ci_lo, t.ci_hi, color=NULL_FILL, linewidth=5, zorder=1)
    ax.scatter(t.completeness, y, s=46, color=[MUTED if m == "lookup table" else OBSERVED for m in t.model], zorder=2)
    ax.set_yticks(y, t.model, fontsize=9, color=INK)
    ax.set_xlabel("completeness: share of the lookup table's gain over the base rate", fontsize=9, color=MUTED)
    ax.set_title("How much of who-answers-whom each theory explains", fontsize=10, color=INK, loc="left")
    _clean(ax).spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bystander(curve, path):
    """One panel per era. Two series: does each agent get lazier, and does the group still answer?"""
    sizes = ["2-4", "5-8", "9-15", "16+"]
    eras = sorted(curve.era.unique())
    if not eras:  # no broadcast requests in this transcript
        return
    fig, axes = plt.subplots(1, len(eras), figsize=(3.3 * len(eras) + 0.6, 3.5), sharey=True, squeeze=False)
    for ax, era in zip(axes[0], eras):
        c = curve[curve.era == era].set_index("size").reindex(sizes)
        x = np.arange(len(sizes))
        for col, colour, label in [("p_any_agent_responds", SERIES[0], "any agent answers"), ("p_each_agent_responds", SERIES[1], "a given agent answers")]:
            ok = c[col].notna().to_numpy()
            ax.plot(x[ok], c[col][ok], color=colour, linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
            for xi, v in zip(x[ok], c[col][ok]):
                ax.annotate(f"{v:.0%}", (xi, v), fontsize=7.5, color=INK, ha="center", xytext=(0, 9), textcoords="offset points")
        ax.set_xticks(x, sizes)
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(f"era {era}", fontsize=10, color=INK, loc="left")
        _clean(ax)
    axes[0][len(eras) // 2].set_xlabel("agents active in the room that day", fontsize=9, color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Bystander effect on broadcast requests, within era (group size and era are confounded across eras)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hierarchy(table, path, statistic="steepness"):
    """Observed steepness against its null band, one row per group."""
    t = table[table.statistic == statistic].dropna(subset=["null_lo"])
    if t.empty:
        return
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(t) + 1.6))
    ax.hlines(y, t.null_lo, t.null_hi, color=NULL_FILL, linewidth=6, zorder=1, label="null 95% band")
    ax.scatter(t.null_mean, y, s=18, color=MUTED, zorder=2, label="null mean")
    ax.scatter(t.observed, y, s=52, color=OBSERVED, zorder=3, label="observed")
    for yi, r in zip(y, t.itertuples()):
        ax.annotate("p < 0.002" if r.p < 0.002 else f"p = {r.p:.3f}", (max(r.observed, r.null_hi), yi), fontsize=7.5, color=MUTED, xytext=(8, -3), textcoords="offset points")
    ax.set_yticks(y, t.group, fontsize=9, color=INK)
    ax.set_xlabel(f"{statistic.replace('_', ' ')} of the dominance hierarchy", fontsize=9, color=MUTED)
    ax.set_title("Is the hierarchy steeper than chance?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    _clean(ax).spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def trends(weekly, series, era_starts, path):
    """Stacked weekly series sharing one time axis, era boundaries marked on every panel."""
    fig, axes = plt.subplots(len(series), 1, figsize=(8.6, 1.45 * len(series) + 0.8), sharex=True)
    for ax, (col, label) in zip(axes, series):
        ax.plot(weekly.week, weekly[col], color=INK, linewidth=1.3)
        for era, start in era_starts.items():
            ax.axvline(start, color=OBSERVED, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
            if ax is axes[0]:
                ax.annotate(f"era {era}", (start, 0.86), xycoords=("data", "axes fraction"), fontsize=8, color=MUTED, xytext=(4, 0), textcoords="offset points")
        ax.set_title(label, fontsize=8.5, color=INK, loc="left", pad=3)
        ax.set_ylim(bottom=0)
        if weekly[col].max() <= 1:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.grid(axis="y", color=FAINT, linewidth=0.6)
        ax.set_axisbelow(True)
        _clean(ax)
    fig.suptitle("Weekly series behind the pooled numbers (dashed lines: era boundaries)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def network_timeline(edges, agents, path, title, freq="Q", top_edges=70, ncols=3):
    """The same network, one panel per period, every agent at a fixed place on a circle.

    Agents sit in order of joining (clockwise from the top), so cohorts are arcs and
    a tie across the circle is a tie between old and new. `edges` has ts, source,
    target, weight; only agents with a tie in the period are drawn.
    """
    order = agents.sort_values("joined").agent.tolist()
    angle = {a: np.pi / 2 - 2 * np.pi * i / len(order) for i, a in enumerate(order)}
    xy = {a: np.array([np.cos(t), np.sin(t)]) for a, t in angle.items()}
    fam = dict(zip(agents.agent, agents.family))
    e = edges[edges.source.isin(xy) & edges.target.isin(xy) & (edges.source != edges.target)]
    e = e.assign(period=pd.to_datetime(e.ts).dt.to_period(freq))
    periods = sorted(e.period.unique())
    nrows = int(np.ceil(len(periods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
    for ax, per in zip(axes.flat, periods):
        g = e[e.period == per].groupby(["source", "target"]).weight.sum().reset_index()
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=FAINT, linewidth=0.8))
        strongest = g.nlargest(top_edges, "weight")
        wmax = strongest.weight.max()
        for r in strongest.itertuples():
            a, b = xy[r.source], xy[r.target]
            mid = (a + b) / 2 * 0.35  # pull the chord toward the centre so ties between neighbours stay visible
            t = np.linspace(0, 1, 30)[:, None]
            curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t**2 * b
            ax.plot(curve[:, 0], curve[:, 1], color=INK, alpha=0.10 + 0.45 * r.weight / wmax, linewidth=0.4 + 2.6 * r.weight / wmax, solid_capstyle="round")
        ins = g.groupby("target").weight.sum()
        active = sorted(set(g.source) | set(g.target), key=order.index)
        for a in active:
            ax.scatter(*xy[a], s=30 + 260 * ins.get(a, 0) / ins.max(), color=family_colour(fam.get(a)), edgecolors="white", linewidths=0.8, zorder=3)
            deg = np.degrees(angle[a]) % 360
            flip = 90 < deg < 270
            ax.text(*(xy[a] * 1.09), a, fontsize=5.8, color=INK, rotation=deg + 180 if flip else deg, rotation_mode="anchor", ha="right" if flip else "left", va="center")
        ax.set_title(f"{per}   {len(active)} agents", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c) for c in SERIES + [OTHER]]
    fig.legend(handles, NAMED_FAMILIES + ["other"], frameon=False, fontsize=9, labelcolor=INK, loc="upper right", ncol=4, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
