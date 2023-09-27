import { Matrix } from 'ml-matrix';
import { expect, it, describe } from 'vitest';

import fcnnlsVector from '../fcnnlsVector';

import { assertResult } from './assertResult';
import { data } from './data/convergence';
import { data as data1 } from './deconv-examples/single_shifted';

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
    const result = fcnnlsVector(X, y, {
      maxIterations: 1000,
      gradientTolerance: 1e-10,
    });
    assertResult(result, Matrix.columnVector(solution), 4);
  });

  it('negative identity X and positive RHS', () => {
    const X = Matrix.eye(3).mul(-1);
    const y = [1, 2, 3];
    const solution = [0, 0, 0];
    const result = fcnnlsVector(X, y).to1DArray();
    expect(result).toEqual(solution);
  });
  it('Convergence for tricky case', () => {
    const result = fcnnlsVector(Matrix.checkMatrix(data.mC), data.bf);
    const solution = [0, 50, 10];
    assertResult(result, Matrix.columnVector(solution), 4);
  });
  it('Convergence where intercept is not zero', () => {
    const result = fcnnlsVector(data1.X, data1.Y, {
      interceptAtZero: false,
      gradientTolerance: 1e-10,
      maxIterations: 1000,
    });
    const result2 = fcnnlsVector(data1.X5, data1.Y, { interceptAtZero: true });
    const scipyResult = [4.92128988, 0.34302285, 0.58189576];
    assertResult(result, Matrix.columnVector(scipyResult), 4);
    expect(result).toStrictEqual(result2);
  });
});
