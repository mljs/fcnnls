import { Matrix } from 'ml-matrix';
import { expect, it, describe } from 'vitest';

import { fcnnlsVector } from '../fcnnlsVector';

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
    const result = fcnnlsVector(X, y);
    assertResult(result, Matrix.columnVector(solution), 4);
  });
  it('Convergence for tricky case', () => {
    const result = fcnnlsVector(Matrix.checkMatrix(data.mC), data.bf);
    const solution = [0, 50, 10];
    assertResult(result, Matrix.columnVector(solution), 4);
  });
  it('intercept not forced to zero', () => {
    const result = fcnnlsVector(data1.X, data1.Y, {
      interceptAtZero: false,
      gradientTolerance: 1e-10,
    });
    const result2 = fcnnlsVector(data1.X5, data1.Y, { interceptAtZero: true });
    expect(result).toStrictEqual(result2);

    const scipyResult = [4.92128988, 0.34302285, 0.58189576];
    assertResult(result, Matrix.columnVector(scipyResult), 4);

    const scipyUnshiftedResult = [0, 0.93375969];
    const scipyError = 11.562826502843999;
    const resultUnshifted = fcnnlsVector(data1.X, data1.Y, {
      info: true,
    });
    assertResult(resultUnshifted, Matrix.columnVector(scipyUnshiftedResult), 4);
    expect(resultUnshifted.info.rse[1][0]).toBeCloseTo(scipyError, 4);
  });

  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = [0, 1, 2, 3];
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = fcnnlsVector(X, Y);
    assertResult(result, solution);
  });

  it('simple case', () => {
    const X = new Matrix([
      [1, 0],
      [2, 0],
      [3, 0],
      [0, 1],
    ]);
    const Y = [1, 2, 3, 4];
    const solution = new Matrix([[1], [4]]);
    const result = fcnnlsVector(X, Y, { info: true });
    assertResult(result, solution);
    expect(result.info.rse).toStrictEqual([[0]]);
  });

  it('simple case that requires a negative coefficient', () => {
    const X = new Matrix([
      [1, 0],
      [2, 0],
      [3, 0],
      [0, -1],
    ]);
    const Y = [1, 2, 3, 4];
    const solution = new Matrix([[1], [0]]);
    const result = fcnnlsVector(X, Y, { info: true });
    assertResult(result, solution);
    expect(result.info.rse).toStrictEqual([[4], [4]]);
  });

  it('non-singular square X, Y 3x1', () => {
    const X = new Matrix([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
    const Y = [-1, 2, -3];
    const solution = new Matrix([[0], [0], [0.5]]);
    const result = fcnnlsVector(X, Y);
    assertResult(result, solution);
  });

  it('singular square X rank 2, Y 3x1', () => {
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const Y = [-1, 0, 10];
    const solution = new Matrix([[1.0455], [0], [0]]);
    const result = fcnnlsVector(X, Y);
    assertResult(result, solution);
  });

  it('non positive-definite matrix', () => {
    const X = new Matrix([
      [1, 1, 1, 0],
      [0, 1, 1, 1],
      [1, 2, 2, 1],
    ]);
    const Y = [-2, 2, 0];
    const solution = new Matrix([[0], [0], [0], [1]]);
    const result = fcnnlsVector(X, Y);
    assertResult(result, solution);
  });
  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = [0, 1, 2, 3];
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = fcnnlsVector(X, Y);
    assertResult(result, solution);
  });
});
