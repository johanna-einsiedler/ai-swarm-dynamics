"""Network statistics with pre-network permutation nulls (Bejder et al. 1998; Farine 2017).

The raw event stream is permuted, never the finished network, and every
statistic is recomputed on each permuted stream.

  speaker   shuffle who sent each message, within room x day.
            Keeps each agent's daily message count per room and every
            message's targets; breaks who-addresses-whom.
  target    shuffle the targets among mentions, within room x day.
            Keeps each agent's daily out-volume and in-volume; breaks
            dyadic preference only.

Both nulls keep every agent's in- and out-volume per room-day, so volume
statistics (degree, centralisation) are fixed by construction and are reported
without a null.
"""
import numpy as np
import pandas as pd


# ---------- statistics on a weighted directed matrix W[i, j] = weight of i -> j

def reciprocity(W):
    """Share of all weight that is matched in the opposite direction."""
    total = W.sum()
    return 2 * np.minimum(W, W.T)[np.triu_indices_from(W, 1)].sum() / total if total else np.nan


def same_family_share(W, fam):
    """Share of weight between agents of the same model family."""
    total = W.sum()
    return W[fam[:, None] == fam[None, :]].sum() / total if total else np.nan


def partner_selectivity(W):
    """Mean concentration of each agent's out-weight over partners (Herfindahl, 1/k = even)."""
    out = W.sum(1)
    keep = out > 0
    if not keep.any():
        return np.nan
    P = W[keep] / out[keep, None]
    return float((P**2).sum(1).mean())


def transitivity(W):
    """Share of connected triples that close, on ties stronger than the median dyad."""
    S = W + W.T
    nz = S[np.triu_indices_from(S, 1)]
    nz = nz[nz > 0]
    if len(nz) < 3:
        return np.nan
    A = (S > np.median(nz)).astype(float)
    np.fill_diagonal(A, 0)
    k = A.sum(1)
    triples = (k * (k - 1)).sum() / 2
    return float(np.trace(A @ A @ A) / 6 * 3 / triples) if triples else np.nan


def in_strength_gini(W):
    x = np.sort(W.sum(0))
    x = x[x > 0]
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n else np.nan


STATS = {
    "reciprocity": lambda W, fam: reciprocity(W),
    "same_family_share": same_family_share,
    "partner_selectivity": lambda W, fam: partner_selectivity(W),
    "transitivity": lambda W, fam: transitivity(W),
}
DESCRIPTIVE = {"in_strength_gini": lambda W, fam: in_strength_gini(W)}


# ---------- permutation machinery

class MentionStream:
    """Agent-to-agent mentions as integer arrays, ready for fast permutation."""

    def __init__(self, events, mentions, agents, group="era", kinds=("at", "name")):
        msgs = events[(events.type == "message") & (events.actor_class == "agent")].reset_index(drop=True)
        self.names = sorted(agents.agent)
        code = {a: i for i, a in enumerate(self.names)}
        self.n = len(self.names)
        self.fam = agents.set_index("agent").family.reindex(self.names).to_numpy()
        self.msg_actor = msgs.actor.map(code).to_numpy()
        self.msg_block = pd.factorize(msgs.room + "|" + msgs.day)[0]
        self.groups, gcodes = np.unique(msgs[group].to_numpy(), return_inverse=True)
        msg_idx = pd.Series(np.arange(len(msgs)), index=msgs.event_id)
        m = mentions[(mentions.actor_class == "agent") & mentions.kind.isin(kinds)]
        self.m_msg = msg_idx.reindex(m.event_id).to_numpy()
        self.m_target = m.target.map(code).to_numpy()
        self.m_weight = m.weight.to_numpy()
        self.m_group = gcodes[self.m_msg]
        self.m_block = self.msg_block[self.m_msg]

    def matrices(self, actor, target):
        """-> W[group, i, j], self-loops dropped."""
        n, g = self.n, len(self.groups)
        keep = actor != target
        idx = (self.m_group[keep] * n + actor[keep]) * n + target[keep]
        return np.bincount(idx, weights=self.m_weight[keep], minlength=g * n * n).reshape(g, n, n)

    def observed(self):
        return self.matrices(self.msg_actor[self.m_msg], self.m_target)

    @staticmethod
    def _shuffle_within(values, blocks, rng):
        order = np.lexsort((rng.random(len(values)), blocks))
        out = np.empty_like(values)
        out[np.argsort(blocks, kind="stable")] = values[order]
        return out

    def permuted(self, null, rng):
        if null == "speaker":
            actor = self._shuffle_within(self.msg_actor, self.msg_block, rng)[self.m_msg]
            return self.matrices(actor, self.m_target)
        if null == "target":
            target = self._shuffle_within(self.m_target, self.m_block, rng)
            return self.matrices(self.msg_actor[self.m_msg], target)
        raise ValueError(null)


def run(stream, nulls=("speaker", "target"), n_perm=1000, seed=0):
    """-> (summary table, {(null, stat, group): null draws})"""
    rng = np.random.default_rng(seed)
    obs = stream.observed()
    rows, draws = [], {}
    for null in nulls:
        sims = {s: np.empty((n_perm, len(stream.groups))) for s in STATS}
        for p in range(n_perm):
            Wp = stream.permuted(null, rng)
            for s, f in STATS.items():
                for g in range(len(stream.groups)):
                    sims[s][p, g] = f(Wp[g], stream.fam)
        for s, f in STATS.items():
            for g, name in enumerate(stream.groups):
                o = f(obs[g], stream.fam)
                d = sims[s][:, g]
                d = d[~np.isnan(d)]
                draws[(null, s, name)] = d
                hi, lo = (d >= o).sum(), (d <= o).sum()
                p_val = min(1.0, 2 * (min(hi, lo) + 1) / (len(d) + 1)) if len(d) else np.nan
                rows.append((s, name, null, o, d.mean(), np.quantile(d, 0.025), np.quantile(d, 0.975), p_val))
    for s, f in DESCRIPTIVE.items():
        for g, name in enumerate(stream.groups):
            rows.append((s, name, "none", f(obs[g], stream.fam), np.nan, np.nan, np.nan, np.nan))
    table = pd.DataFrame(rows, columns=["statistic", "group", "null", "observed", "null_mean", "null_lo", "null_hi", "p"])
    return table, draws, obs
