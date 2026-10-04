"""`swarm-sna report`: the fixed battery, each statistic with its null distribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, nulls

LABELS = {
    "reciprocity": "Reciprocity",
    "same_family_share": "Same-family share",
    "partner_selectivity": "Partner selectivity",
    "transitivity": "Transitivity",
    "in_strength_gini": "In-strength Gini",
}
INK, MUTED, NULL_FILL, OBSERVED = "#1f2328", "#6b7280", "#c9ced6", "#c2410c"


def half_weight_index(events):
    """Association: HWI = x / (x + (y_i + y_j) / 2), over room-days. -> long table per group"""
    m = events[(events.type == "message") & (events.actor_class == "agent")]
    pres = m[["era", "room", "day", "actor"]].drop_duplicates()
    rows = []
    for era, g in pres.groupby("era"):
        g = g.assign(rd=g.room + "|" + g.day)
        inc = pd.crosstab(g.actor, g.rd).clip(upper=1)
        x = inc.to_numpy() @ inc.to_numpy().T
        days = np.diag(x)
        hwi = x / (0.5 * (days[:, None] + days[None, :]))
        iu = np.triu_indices(len(inc), 1)
        rows.append(pd.DataFrame({"era": era, "a": inc.index[iu[0]], "b": inc.index[iu[1]], "together": x[iu], "hwi": hwi[iu]}))
    return pd.concat(rows, ignore_index=True)


def figure(table, draws, groups, path, null="speaker"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = list(nulls.STATS)
    fig, axes = plt.subplots(len(stats), len(groups), figsize=(max(7.5, 3.4 * len(groups)), 2.0 * len(stats)), squeeze=False)
    for r, s in enumerate(stats):
        for c, g in enumerate(groups):
            ax = axes[r][c]
            d = draws[(null, s, g)]
            row = table[(table.statistic == s) & (table.group == g) & (table.null == null)].iloc[0]
            lo, hi = min(d.min(), row.observed), max(d.max(), row.observed)
            pad = (hi - lo) * 0.08 or 0.01
            ax.hist(d, bins=30, color=NULL_FILL, edgecolor=NULL_FILL, linewidth=0.6)
            ax.axvline(row.observed, color=OBSERVED, linewidth=2)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.9)  # headroom for the annotation
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(axis="x", colors=MUTED, labelsize=8)
            p_txt = "p < 0.002" if row.p < 0.002 else f"p = {row.p:.3f}"
            side = "left" if row.observed > d.mean() else "right"  # text goes opposite the observed line
            ax.text(0.98 if side == "right" else 0.02, 0.92, f"observed {row.observed:.3f}\nnull {row.null_mean:.3f}\n{p_txt}", transform=ax.transAxes, ha=side, va="top", fontsize=8, color=INK)
            if r == 0:
                ax.set_title(f"{g}", fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(LABELS[s], fontsize=9, color=INK)
    fig.suptitle(f"Observed value (orange line) against {len(next(iter(draws.values())))} permutations of the event stream (grey), {null}-swap null", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(in_dir, n_perm=1000, group="era", seed=0, kinds=("at", "name")):
    in_dir = Path(in_dir)
    events = pd.read_parquet(in_dir / "events.parquet")
    mentions = pd.read_parquet(in_dir / "mentions.parquet")
    agents = pd.read_parquet(in_dir / "agents.parquet")
    stream = nulls.MentionStream(events, mentions, agents, group=group, kinds=kinds)
    table, draws, obs = nulls.run(stream, n_perm=n_perm, seed=seed)
    table["group"] = [f"{group} {g}" for g in table.group]
    draws = {(n, s, f"{group} {g}"): d for (n, s, g), d in draws.items()}
    groups = [f"{group} {g}" for g in stream.groups]

    table["permutations"] = n_perm
    table.to_csv(in_dir / f"report_{group}.csv", index=False)
    figure(table, draws, groups, in_dir / f"report_{group}_nulls.png")
    # The null draws themselves, for the report card's histograms.
    nested = {}
    for (null, stat, g), d in draws.items():
        nested.setdefault(null, {}).setdefault(stat, {})[g] = [round(float(v), 4) for v in d]
    (in_dir / f"report_{group}_draws.json").write_text(json.dumps(nested))
    figures.networks(obs, stream.names, dict(zip(agents.agent, agents.family)), groups, in_dir / f"network_{group}.png")
    hwi = half_weight_index(events)
    hwi.to_csv(in_dir / "association_hwi.csv", index=False)

    sizes = pd.DataFrame({"group": groups, "agents": [(W.sum(0) + W.sum(1) > 0).sum() for W in obs], "mention_weight": [W.sum() for W in obs]})
    print(f"\nmention network per {group} ({n_perm} permutations per null)")
    print(sizes.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    for null in ("speaker", "target"):
        t = table[table.null == null]
        print(f"\n{null}-swap null (within room x day)")
        print(f"{'statistic':22s}{'group':8s}{'observed':>10s}{'null mean':>11s}{'null 95% band':>19s}{'p':>9s}")
        for r in t.itertuples():
            p_txt = "<0.002" if r.p < 0.002 else f"{r.p:.3f}"
            print(f"{LABELS[r.statistic]:22s}{r.group:8s}{r.observed:10.3f}{r.null_mean:11.3f}{f'[{r.null_lo:.3f}, {r.null_hi:.3f}]':>19s}{p_txt:>9s}")
    t = table[table.null == "none"]
    print("\nno null (fixed by both permutations)")
    for r in t.itertuples():
        print(f"{LABELS[r.statistic]:22s}{r.group:8s}{r.observed:10.3f}")
    print("\nassociation (half-weight index over room-days), descriptive")
    print(hwi.groupby("era").hwi.describe()[["count", "mean", "50%", "max"]].round(3).to_string())
    print(f"\nwrote {in_dir}/report_{group}.csv, report_{group}_nulls.png, network_{group}.png, association_hwi.csv")
