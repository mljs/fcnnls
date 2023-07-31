import { Matrix } from 'ml-matrix';
import { expect, it, describe } from 'vitest';

import fcnnlsVector from '../fcnnlsVector';

import { data } from './data/convergence';

describe('Test single right hand side convergence', () => {
  it('example documentation', () => {
    const X = new Matrix([
      [1, 1, 2],
      [10, 11, -9],
      [-1, 0, 0],
      [-5, 6, -7],
    ]);
    const y = [-1, 11, 0, 1];
    const solution = [0.461, 0.5611, 0];
    const result = fcnnlsVector(X, y);
    for (const r of result) {
      expect(r).toBeCloseTo(solution.shift() as number, 4);
    }
  });

  it('negative identity X and positive RHS', () => {
    const X = Matrix.eye(3).mul(-1);
    const y = [1, 2, 3];
    const solution = [0, 0, 0];
    const result = fcnnlsVector(X, y);
    expect(result).toEqual(solution);
  });
  it('Convergence for tricky case', () => {
    const result = fcnnlsVector(Matrix.checkMatrix(data.mC), data.bf);
    const solution = [0, 50, 10];
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeCloseTo(solution[i], 8);
    }
  });
});
