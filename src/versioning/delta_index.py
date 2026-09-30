"""Version-delta vectors: one base vector per lineage plus a small residual per distinct version.

EXPERIMENT, off by default. Nothing in the served index, the CLI or the API uses it; it is built from an
existing versioned index's embeddings and measured by `src/bench_real_history.py --delta-vectors`. No
result is claimed for it.

The idea. A lineage's versions are near-duplicates, so their embeddings sit close together. Instead of
storing every version's full vector, store

    base      the mean of the lineage's distinct-content vectors      (L x dim, fp16)
    residual  entry - base, quantised to int8 with one scale per entry  (M x dim, int8)

where an "entry" is a distinct (lineage, content hash): versions with identical normalised text share one
residual, exactly as the content-hash index shares one embedding. Search is two-stage:

    1. rank LINEAGES by cosine between the query and the base vectors, keep the best `lineage_k`;
    2. rebuild the full vectors (base + dequantised residual) of only those lineages' versions and rank
       THEM by cosine.

What it can and cannot win. The saving is in storage: int8 residuals for distinct entries instead of fp16
vectors for every row, roughly `M/N * 0.5` of the embedding matrix plus the small base table. The cost is
(a) int8 quantisation noise in the reconstructed vectors and (b) stage 1 can drop a lineage whose base
vector ranks below `lineage_k` although one of its versions would have scored higher. Whether either
matters is an empirical question the benchmark answers; this module only makes it askable.
"""
from pathlib import Path

import numpy as np

FILE = "delta_vectors.npz"


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class DeltaVectors:
    def __init__(self, lineages, base, entry_lineage, residual, scale, row_entry):
        self.lineages = list(lineages)
        self.base = np.asarray(base, dtype=np.float16)                  # (L, dim)
        self.entry_lineage = np.asarray(entry_lineage, dtype=np.int32)  # (M,)
        self.residual = np.asarray(residual, dtype=np.int8)             # (M, dim)
        self.scale = np.asarray(scale, dtype=np.float32)                # (M,)
        self.row_entry = np.asarray(row_entry, dtype=np.int32)          # (N,)
        self.row_lineage = self.entry_lineage[self.row_entry]
        self._base_unit = _unit(self.base.astype(np.float32))

    # --- build / persist -----------------------------------------------------------------------------

    @classmethod
    def build(cls, embeddings, lineage_of_row, hash_of_row):
        """From the (N, dim) embeddings of a versioned index and, per row, its lineage id and content
        hash. Rows with the same (lineage, hash) share one entry."""
        E = np.asarray(embeddings, dtype=np.float32)
        n, dim = E.shape
        if len(lineage_of_row) != n or len(hash_of_row) != n:
            raise ValueError("lineage_of_row and hash_of_row need one value per embedding row")
        lineages, lidx = [], {}
        entries, entry_of, first_row = {}, [], []
        for i in range(n):
            lid = lineage_of_row[i]
            if lid not in lidx:
                lidx[lid] = len(lineages)
                lineages.append(lid)
            key = (lidx[lid], hash_of_row[i])
            if key not in entries:
                entries[key] = len(entries)
                first_row.append(i)
            entry_of.append(entries[key])
        entry_lineage = np.array([k[0] for k in entries], dtype=np.int32)
        vecs = E[np.array(first_row, dtype=np.int64)]                    # (M, dim), one per entry
        base = np.zeros((len(lineages), dim), dtype=np.float32)
        for li in range(len(lineages)):
            base[li] = vecs[entry_lineage == li].mean(axis=0)
        base16 = base.astype(np.float16)
        res = vecs - base16.astype(np.float32)[entry_lineage]           # against the STORED base
        peak = np.abs(res).max(axis=1)
        scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(res / scale[:, None]), -127, 127).astype(np.int8)
        return cls(lineages, base16, entry_lineage, q, scale, np.array(entry_of, dtype=np.int32))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / FILE, lineages=np.array(self.lineages, dtype=str), base=self.base,
                 entry_lineage=self.entry_lineage, residual=self.residual, scale=self.scale,
                 row_entry=self.row_entry)
        return directory / FILE

    @classmethod
    def load(cls, directory):
        z = np.load(Path(directory) / FILE, allow_pickle=False)
        return cls([str(x) for x in z["lineages"]], z["base"], z["entry_lineage"], z["residual"],
                   z["scale"], z["row_entry"])

    # --- sizes -----------------------------------------------------------------------------------------

    @property
    def n_rows(self):
        return int(self.row_entry.shape[0])

    @property
    def n_entries(self):
        return int(self.residual.shape[0])

    def size_bytes(self):
        """Bytes of the arrays a served delta index would need (lineage ids/texts excluded: the
        versioned index already stores those, so they cost the same either way)."""
        return int(self.base.nbytes + self.entry_lineage.nbytes + self.residual.nbytes
                   + self.scale.nbytes + self.row_entry.nbytes)

    def baseline_bytes(self):
        """What the current approach stores for the same rows: one fp16 vector per row."""
        return int(self.n_rows * self.base.shape[1] * 2)

    # --- vectors and search ------------------------------------------------------------------------------

    def reconstruct(self, entries):
        """Full float32 vectors for entry indices."""
        entries = np.asarray(entries, dtype=np.int64)
        return (self.base[self.entry_lineage[entries]].astype(np.float32)
                + self.residual[entries].astype(np.float32) * self.scale[entries][:, None])

    def full_vectors(self):
        """Reconstructed vector for every ROW (N, dim) -- for tests and error measurement."""
        return self.reconstruct(self.row_entry)

    def search(self, qvec, k=10, lineage_k=None):
        """[(row, score)] best first: lineages by base vector, then their versions by full vector."""
        q = _unit(np.asarray(qvec, dtype=np.float32).reshape(-1))
        lineage_scores = self._base_unit @ q
        n_l = len(lineage_scores)
        keep = min(n_l, int(lineage_k) if lineage_k else max(3 * int(k), 20))
        top = np.argpartition(-lineage_scores, keep - 1)[:keep] if keep < n_l else np.arange(n_l)
        rows = np.nonzero(np.isin(self.row_lineage, top))[0]
        if rows.size == 0:
            return []
        entries, inverse = np.unique(self.row_entry[rows], return_inverse=True)
        scores = (self.reconstruct(entries) @ q)[inverse]
        order = np.argsort(-scores, kind="stable")[:int(k)]
        return [(int(rows[o]), float(scores[o])) for o in order]
