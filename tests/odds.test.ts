import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { americanOdds, bucket, defaultLine, probOver, type ResidualModel } from '../src/odds';

const model = JSON.parse(readFileSync(new URL('../public/data/model.json', import.meta.url), 'utf8')) as {
  residuals: Record<string, ResidualModel>;
};

describe('probOver', () => {
  // Reference values from prob_over() in pipeline/train.py on the same model.json
  it.each([
    ['pts', 22.4, 20.5, 0.5918367346938775],
    ['pts', 8.1, 9.5, 0.32653061224489793],
    ['reb', 6.3, 5.5, 0.5714285714285714],
    ['fg3', 1.2, 1.5, 0.3673469387755102],
  ])('%s projected %f vs line %f matches Python', (stat, pred, line, expected) => {
    expect(probOver(pred, line, model.residuals[stat])).toBeCloseTo(expected, 10);
  });

  it('goes down as the line goes up', () => {
    const rm = model.residuals.pts;
    expect(probOver(20, 15.5, rm)).toBeGreaterThan(probOver(20, 20.5, rm));
    expect(probOver(20, 20.5, rm)).toBeGreaterThan(probOver(20, 25.5, rm));
  });

  it('puts projections in the right bucket', () => {
    const rm: ResidualModel = { edges: [null, 5, 10, null], quantiles: [[0], [0], [0]] };
    expect(bucket(2, rm)).toBe(0);
    expect(bucket(5, rm)).toBe(1);
    expect(bucket(50, rm)).toBe(2);
  });
});

describe('lines and odds', () => {
  it('rounds the default line to the .5 below the average', () => {
    expect(defaultLine(24.7)).toBe(24.5);
    expect(defaultLine(3)).toBe(3.5);
  });

  it('converts probabilities to American odds', () => {
    expect(americanOdds(0.6)).toBe('-150');
    expect(americanOdds(0.4)).toBe('+150');
    expect(americanOdds(0.5)).toBe('-100');
  });
});
