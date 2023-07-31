import { Matrix } from 'ml-matrix';
import { it, describe } from 'vitest';

import { cssls } from '../cssls';
import { initialisation } from '../initialisation';

import { assertResult } from './fcnnls.test';

describe('cssls test', () => {
  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = cssls(XtX, XtY, Pset, l, p);
    assertResult(result, solution);
  });
  it('identity X, Y 5x3', () => {
    const X = Matrix.eye(5);
    const Y = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    const init = initialisation(X, Y);
    const solution = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    const result = cssls(init.XtX, init.XtY, init.Pset, init.l, init.p);
    assertResult(result, solution);
  });
  it('non-singular square X, Y 3x1', () => {
    const X = new Matrix([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
    const Y = new Matrix([[-1], [2], [-3]]);
    const { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    const solution = new Matrix([[-1], [0], [1]]);
    const result = cssls(XtX, XtY, Pset, l, p);
    assertResult(result, solution);
  });
  it('ill-conditionned square X rank 2, Y 3x1', () => {
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    const Y = new Matrix([[-1], [0], [10]]);
    const { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    const solution = new Matrix([[1.0455], [0], [0]]);
    const result = cssls(XtX, XtY, Pset, l, p);
    assertResult(result, solution);
  });
  it('6x3 X full-rank, Y 6x7', () => {
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [0, 1, 1],
    ]);
    const Y = new Matrix([
      [-1, 0, 1, 2, 3, 4, 5],
      [0, 3, 5, 6, 79, 3, 1],
      [10, 11, 2, 3, 4, 7, 8],
      [1, 112, 0, 0, 0, 7, 8],
      [1000, 2, 56, 40, 1, 1, 3],
      [7, 6, 5, 4, 3, 2, 1],
    ]);
    const { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    const solution = new Matrix([
      [203.7567, 0, 0, 0, 0, 0, 0],
      [-149.1338, 3.3309, 2.0243, 1.5134, 0, 0, 0],
      [0, 0, 0, 0, 1.0827, 0.3911, 0.4738],
    ]);
    const result = Matrix.round(cssls(XtX, XtY, Pset, l, p).mul(10000)).mul(
      0.0001,
    );
    assertResult(result, solution);
  });
  it('Van Benthem - Keenan example', () => {
    const X = new Matrix([
      [95, 89, 82],
      [23, 76, 44],
      [61, 46, 62],
      [42, 2, 79],
    ]);
    const Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);

    const { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    const solution = new Matrix([
      [0, 0.6873, 0.2836],
      [0.6272, 0, 0.2862],
      [0.3517, 0.2873, 0.335],
    ]);
    const result = Matrix.round(cssls(XtX, XtY, Pset, l, p).mul(10000)).mul(
      0.0001,
    );
    assertResult(result, solution);
  });
  it('negative identity X, positive Y', () => {
    const X = Matrix.eye(3).mul(-1);
    const Y = new Matrix([[1], [2], [3]]);
    const init = initialisation(X, Y);
    const solution = new Matrix([[-1], [-2], [-3]]);
    const result = cssls(init.XtX, init.XtY, null, init.l, init.p);
    assertResult(result, solution);
  });

  it('non positive-definite matrix', () => {
    const X = new Matrix([
      [1, 1, 1, 0],
      [0, 1, 1, 1],
      [1, 2, 2, 1],
    ]);
    const Y = new Matrix([[-2], [2], [0]]);
    const init = initialisation(X, Y);
    const solution = new Matrix([[-2], [0], [0], [2]]);
    const result = cssls(init.XtX, init.XtY, null, init.l, init.p);
    assertResult(result, solution);
  });

  it('non positive-definite matrix with Pset', () => {
    const X = new Matrix([
      [1, 1, 1, 0],
      [0, 1, 1, 1],
      [1, 2, 2, 1],
    ]);
    const Y = new Matrix([[-2], [2], [0]]);
    const init = initialisation(X, Y);
    const solution = new Matrix([[0], [0], [0], [1]]);
    const result = cssls(init.XtX, init.XtY, init.Pset, init.l, init.p);
    assertResult(result, solution);
  });

  it('big low-rank matrix 10x9 X', () => {
    const X = new Matrix([
      [1, 1, 1, 0, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 0, 0, 0, 1, 1],
      [1, 1, 1, 1, 0, 0, 1, 0, 0],
      [1, 1, 1, 1, 0, 0, 1, 0, 0],
      [2, 2, 2, 0, 2, 2, 2, 2, 0],
      [1, 2, 2, 1, 1, 1, 1, 2, 0],
      [0, 5, 5, 5, 0, 0, 0, 5, 5],
      [2, 2, 2, 0, 2, 2, 2, 2, 0],
      [11, 11, 11, 0, 11, 11, 11, 11, 0],
      [0, 23, 23, 23, 0, 0, 0, 23, 23],
    ]);
    const Y = new Matrix([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const init = initialisation(X, Y);
    const solution = new Matrix([
      [-3.62e-1],
      [1.16712],
      [1.16712],
      [1.89038],
      [-7.23e-1],
      [-7.23e-1],
      [-3.62e-1],
      [0.806154],
      [-4.54969],
    ]);
    const result = cssls(init.XtX, init.XtY, null, init.l, init.p);
    assertResult(result, solution, 3);
  });
});
