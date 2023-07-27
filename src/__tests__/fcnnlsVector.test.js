import { toBeDeepCloseTo, toMatchCloseTo } from 'jest-matcher-deep-close-to';
import { Matrix } from 'ml-matrix';

import fcnnlsVector from '../fcnnlsVector';

import { data } from './data/convergence';

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

describe('myModule test', () => {
  it('example documentation', () => {
    let X = new Matrix([
      [1, 1, 2],
      [10, 11, -9],
      [-1, 0, 0],
      [-5, 6, -7],
    ]);
    let y = [-1, 11, 0, 1];
    let solution = [0.461, 0.5611, 0];
    let result = fcnnlsVector(X, y);
    expect(result).toBeDeepCloseTo(solution, 4);
  });

  it('negative identity X and positive RHS', () => {
    let X = Matrix.eye(3).mul(-1);
    let y = [1, 2, 3];
    let solution = [0, 0, 0];
    let result = fcnnlsVector(X, y);
    expect(result).toBeDeepCloseTo(solution, 4);
  });
  it('Convergence for tricky case', () => {
    const result = fcnnlsVector(Matrix.checkMatrix(data.mC), data.bf, {
      gradientTolerance: 1e-5,
    });
    const solution = [0, 50, 10];
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeDeepCloseTo(solution[i], 8);
    }
  });
});
