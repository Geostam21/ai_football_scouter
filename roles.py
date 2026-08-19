"""
roles.py — data-driven *playing-style* archetypes.

Position tells you where a player lines up; it does not tell you how they play.
Two strikers can both be "STC" yet be a poacher and a target man — completely
different signings. Following the standard scouting-analytics approach (cluster
players on style-defining attributes, then label the clusters), this module
groups players within each position bucket into style archetypes via K-means.

This mirrors how clubs like Brentford/Brighton and the academic literature
(role clustering + similarity) separate *role* from *quality*: we deliberately
cluster on style traits, not on overall ability, so a cluster contains players
of the same *type* across quality levels — which is what surfaces cheaper
alternatives that play like an expensive target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Position buckets we cluster within. Each maps to the role codes that belong to
# it and the style-defining attributes used for clustering (NOT overall quality
# — we exclude generic "how good" and keep "how they play").
_BUCKETS = {
    "Striker": {
        "codes": {"STC"},
        "attrs": ["Fin", "OtB", "Hea", "Str", "Pac", "Acc", "Dri", "Fir",
                  "Tec", "Fla", "Cmp", "Wor", "Agg", "Jum"],
    },
    "Winger": {
        "codes": {"AMR", "AML", "MR", "ML"},
        "attrs": ["Pac", "Acc", "Dri", "Cro", "Tec", "Fla", "OtB", "Fin",
                  "Agi", "Pas", "Wor", "Sta"],
    },
    "Attacking Mid": {
        "codes": {"AMC"},
        "attrs": ["Pas", "Vis", "Tec", "Dri", "Fla", "OtB", "Fin", "Fir",
                  "Lon", "Cmp", "Fre"],
    },
    "Central Mid": {
        "codes": {"MC"},
        "attrs": ["Pas", "Vis", "Tck", "Tec", "Dri", "Wor", "Sta", "Lon",
                  "Fir", "OtB", "Dec", "Pos"],
    },
    "Defensive Mid": {
        "codes": {"DM"},
        "attrs": ["Tck", "Mar", "Pos", "Str", "Pas", "Wor", "Sta", "Agg",
                  "Ant", "Dec", "Vis"],
    },
    "Centre-back": {
        "codes": {"DC"},
        "attrs": ["Tck", "Mar", "Hea", "Str", "Jum", "Pos", "Pac", "Acc",
                  "Pas", "Bra", "Agg", "Ant", "Cmp"],
    },
    "Full-back": {
        "codes": {"DR", "DL", "WBR", "WBL"},
        "attrs": ["Tck", "Mar", "Pos", "Pac", "Acc", "Cro", "Dri", "Sta",
                  "Wor", "Tec", "Pas", "OtB"],
    },
}

# Human-readable style labels per bucket. Each cluster is matched to a label by
# scoring its centroid on a small "signature" of attributes that defines that
# style, then assigning labels greedily to the best-matching cluster. Using a
# signature per label (not one shared axis) is what makes the mapping correct —
# a target man is defined by heading+strength, a poacher by finishing+movement.
_STYLE_SIGNATURES = {
    "Striker": {
        "Poacher": ["Fin", "OtB", "Acc"],
        "Target Man": ["Hea", "Str", "Jum"],
        "Complete Forward": ["Fin", "Dri", "Tec", "Str"],
        "Pressing Forward": ["Wor", "Agg", "Sta"],
    },
    "Winger": {
        "Inside Forward": ["Fin", "Dri", "Acc"],
        "Classic Winger": ["Cro", "Pac"],
        "Wide Playmaker": ["Pas", "Vis", "Tec"],
        "Work-rate Wideman": ["Wor", "Sta", "Tck"],
    },
    "Attacking Mid": {
        "Advanced Playmaker": ["Pas", "Vis", "Tec"],
        "Shadow Striker": ["Fin", "OtB", "Lon"],
        "Trequartista": ["Fla", "Dri", "Tec"],
    },
    "Central Mid": {
        "Deep-lying Playmaker": ["Pas", "Vis", "Tec"],
        "Box-to-Box": ["Sta", "Wor", "OtB"],
        "Ball-winner": ["Tck", "Mar", "Agg"],
    },
    "Defensive Mid": {
        "Regista": ["Pas", "Vis", "Tec"],
        "Anchor": ["Pos", "Mar", "Cnt"],
        "Ball-winning DM": ["Tck", "Agg", "Str"],
    },
    "Centre-back": {
        "Ball-playing Defender": ["Pas", "Tec", "Cmp"],
        "No-nonsense Stopper": ["Hea", "Str", "Mar"],
        "Pace CB": ["Pac", "Acc"],
    },
    "Full-back": {
        "Attacking Full-back": ["Cro", "Dri", "OtB"],
        "Wing-back": ["Sta", "Wor", "Pac"],
        "Defensive Full-back": ["Tck", "Mar", "Pos"],
    },
}

_STYLE_LABELS = {
    "Striker": ["Poacher", "Target Man", "Complete Forward", "Pressing Forward"],
    "Winger": ["Inside Forward", "Classic Winger", "Wide Playmaker",
               "Work-rate Wideman"],
    "Attacking Mid": ["Advanced Playmaker", "Shadow Striker", "Trequartista"],
    "Central Mid": ["Deep-lying Playmaker", "Box-to-Box", "Ball-winner"],
    "Defensive Mid": ["Regista", "Anchor", "Ball-winning DM"],
    "Centre-back": ["Ball-playing Defender", "No-nonsense Stopper", "Pace CB"],
    "Full-back": ["Attacking Full-back", "Wing-back", "Defensive Full-back"],
}

_N_CLUSTERS = {"Striker": 4, "Winger": 4, "Attacking Mid": 3, "Central Mid": 3,
               "Defensive Mid": 3, "Centre-back": 3, "Full-back": 3}


class RoleClusterer:
    """Assigns each player a playing-style archetype within their position."""

    def __init__(self, players: pd.DataFrame, seed: int = 42):
        self.seed = seed
        self._models = {}      # bucket -> (attrs, means, stds, centroids, labels)
        self.style = pd.Series(index=players.index, dtype=object)
        self._fit(players)

    def _primary_bucket(self, best_pos_codes) -> str | None:
        """Which single bucket a player belongs to, by their FIRST (best) role.

        A player is clustered in exactly one bucket — the one matching their
        primary position — so a striker who can also fill in at centre-back is
        only ever given a striker style, never a defender one.
        """
        if not best_pos_codes:
            return None
        primary = best_pos_codes[0]   # best_pos_codes is ordered, best first
        for bucket, cfg in _BUCKETS.items():
            if primary in cfg["codes"]:
                return bucket
        return None

    def _fit(self, players: pd.DataFrame):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return  # clustering optional; style stays empty

        col = "best_pos_codes" if "best_pos_codes" in players.columns else "positions"
        # assign every player to exactly one bucket up front
        bucket_of = players[col].apply(self._primary_bucket)

        for bucket, cfg in _BUCKETS.items():
            attrs = [a for a in cfg["attrs"] if a in players.columns]
            if len(attrs) < 3:
                continue
            sub = players[bucket_of == bucket]
            if len(sub) < 200:
                continue

            X = sub[attrs].astype(float)
            # Cluster on each player's RELATIVE profile, not raw ratings. Good
            # players score high on everything, so raw attributes cluster by
            # quality, not style (every star lands in one blob). Subtracting each
            # player's own mean keeps only the *shape* — what he's good at
            # relative to himself — which is what actually defines a playing
            # style. This is the standard fix in the role-clustering literature.
            row_mean = X.mean(axis=1)
            Xrel = X.sub(row_mean, axis=0)
            means, stds = Xrel.mean(), Xrel.std().replace(0, 1)
            Xz = ((Xrel - means) / stds).fillna(0).values

            k = _N_CLUSTERS.get(bucket, 3)
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=3)
            clusters = km.fit_predict(Xz)

            # Match clusters to style labels by how strongly each cluster's
            # centroid expresses that style's signature attributes. We score
            # every (cluster, label) pair on the mean z-value of the signature,
            # then assign greedily from the strongest pair down, so each label
            # goes to the cluster that most embodies it and none repeats.
            labels = _STYLE_LABELS[bucket][:k]
            sigs = _STYLE_SIGNATURES[bucket]
            attr_pos = {a: i for i, a in enumerate(attrs)}
            pairs = []
            for cid in range(k):
                centroid = km.cluster_centers_[cid]
                for lab in labels:
                    sig_attrs = [a for a in sigs[lab] if a in attr_pos]
                    score = (np.mean([centroid[attr_pos[a]] for a in sig_attrs])
                             if sig_attrs else 0.0)
                    pairs.append((score, cid, lab))
            pairs.sort(reverse=True)
            cid_to_label, used_cid, used_lab = {}, set(), set()
            for score, cid, lab in pairs:
                if cid in used_cid or lab in used_lab:
                    continue
                cid_to_label[cid] = lab
                used_cid.add(cid)
                used_lab.add(lab)

            self.style.loc[sub.index] = [cid_to_label[int(c)] for c in clusters]
            self._models[bucket] = dict(attrs=attrs, means=means, stds=stds,
                                        km=km, cid_to_label=cid_to_label,
                                        codes=cfg["codes"])

    def attach(self, players: pd.DataFrame) -> pd.DataFrame:
        """Return players with a 'style' column added."""
        players = players.copy()
        players["style"] = self.style.reindex(players.index)
        return players

    def styles_for_position(self, position_codes) -> list[str]:
        """Which style archetypes exist for a requested position."""
        cset = set(position_codes)
        out = []
        for bucket, cfg in _BUCKETS.items():
            if cfg["codes"] & cset:
                k = _N_CLUSTERS.get(bucket, 3)
                out.extend(_STYLE_LABELS[bucket][:k])
        # dedupe preserving order
        seen, res = set(), []
        for s in out:
            if s not in seen:
                seen.add(s)
                res.append(s)
        return res

    def label_of(self, player_row) -> str | None:
        """Style archetype for a single player row (if their position is covered)."""
        idx = player_row.name
        if idx in self.style.index and pd.notna(self.style.get(idx)):
            return self.style.get(idx)
        return None


# natural-language -> style archetype (English / Greek / greeklish)
STYLE_SYNONYMS = {
    "poacher": "Poacher", "fox in the box": "Poacher", "γκολτζης": "Poacher",
    "goal poacher": "Poacher", "goalpoacher": "Poacher",
    "target man": "Target Man", "targetman": "Target Man", "φυσαρμονικα": "Target Man",
    "hold up": "Target Man", "hold-up": "Target Man", "προβολεας": "Target Man",
    "complete forward": "Complete Forward", "complete striker": "Complete Forward",
    "pressing forward": "Pressing Forward", "presser": "Pressing Forward",
    "inside forward": "Inside Forward", "cut inside": "Inside Forward",
    "classic winger": "Classic Winger", "traditional winger": "Classic Winger",
    "wide playmaker": "Wide Playmaker",
    "advanced playmaker": "Advanced Playmaker", "playmaker": "Advanced Playmaker",
    "δεκαρι": "Advanced Playmaker", "shadow striker": "Shadow Striker",
    "trequartista": "Trequartista", "τρεκαρτιστα": "Trequartista",
    "deep lying playmaker": "Deep-lying Playmaker",
    "deep-lying playmaker": "Deep-lying Playmaker", "regista": "Regista",
    "box to box": "Box-to-Box", "box-to-box": "Box-to-Box", "b2b": "Box-to-Box",
    "ball winner": "Ball-winner", "ball-winner": "Ball-winner",
    "ball winning": "Ball-winning DM", "destroyer": "Ball-winning DM",
    "anchor": "Anchor", "anchor man": "Anchor",
    "ball playing defender": "Ball-playing Defender",
    "ball-playing defender": "Ball-playing Defender",
    "no nonsense": "No-nonsense Stopper", "stopper": "No-nonsense Stopper",
    "pace cb": "Pace CB", "quick centre back": "Pace CB",
    "attacking full back": "Attacking Full-back",
    "attacking full-back": "Attacking Full-back", "wing back": "Wing-back",
    "wing-back": "Wing-back", "wingback": "Wing-back",
    "defensive full back": "Defensive Full-back",
    "defensive full-back": "Defensive Full-back",
}


def detect_style(text: str) -> str | None:
    """Find a requested playing-style archetype in free text."""
    low = " " + text.lower() + " "
    # longer phrases first to avoid partial matches
    for phrase in sorted(STYLE_SYNONYMS, key=len, reverse=True):
        if phrase in low:
            return STYLE_SYNONYMS[phrase]
    return None
