/**
 * Over/under probabilities from a projection.
 *
 * The pipeline stores, for each stat, the spread of real outcomes around the model's
 * projections on a held-out season, bucketed by projection size (big scorers vary more).
 * P(over) is the share of those outcomes that would clear the line.
 */
export interface ResidualModel {
  /** Bucket edges by projection; first and last are open-ended (null). */
  edges: (number | null)[];
  /** For each bucket, 49 residual quantiles (actual − projected). */
  quantiles: number[][];
}

export function bucket(pred: number, rm: ResidualModel): number {
  const n = rm.quantiles.length;
  for (let i = 1; i < n; i++) {
    const edge = rm.edges[i];
    if (edge !== null && pred < edge) return i - 1;
  }
  return n - 1;
}

/** Probability the stat lands strictly above `line`. Matches prob_over in pipeline/train.py. */
export function probOver(pred: number, line: number, rm: ResidualModel): number {
  const q = rm.quantiles[bucket(pred, rm)];
  let over = 0;
  for (const r of q) if (pred + r > line) over++;
  return over / q.length;
}

/** Default line: the player's season average rounded down to the nearest .5, like a sportsbook line. */
export const defaultLine = (avg: number) => Math.floor(avg) + 0.5;

/** American odds for a fair price at probability p, e.g. 0.6 → "-150". */
export function americanOdds(p: number): string {
  if (p <= 0 || p >= 1) return '—';
  return p >= 0.5 ? `${Math.round((-100 * p) / (1 - p))}` : `+${Math.round((100 * (1 - p)) / p)}`;
}
