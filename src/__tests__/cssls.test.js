'use strict';

const { Matrix } = require('ml-matrix');
const {
  toBeDeepCloseTo,
  toMatchCloseTo,
} = require('jest-matcher-deep-close-to');

const selection = require('../util/selection');
const cssls = require('../cssls');
const initialisation = require('../initialisation');

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

// to be added: test for svd inverse...
describe('cssls test', () => {
  it('identity X, Y 4x1', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = cssls(XtX, XtY, Pset, l, p);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('identity X, Y 5x3', () => {
    let X = Matrix.eye(5);
    let Y = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    let init = initialisation(X, Y);
    let solution = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    let result = cssls(init.XtX, init.XtY, init.Pset, init.l, init.p);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('non-singular square X, Y 3x1', () => {
    let X = new Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]]);
    let Y = new Matrix([[-1], [2], [-3]]);
    let { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    let solution = new Matrix([[-1], [0], [1]]);
    let result = cssls(XtX, XtY, Pset, l, p);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('ill-conditionned square X rank 2, Y 3x1', () => {
    let X = new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    let Y = new Matrix([[-1], [0], [10]]);
    let { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    let solution = new Matrix([[1.0455], [0], [0]]);
    let result = cssls(XtX, XtY, Pset, l, p);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('6x3 X full-rank, Y 6x7', () => {
    let X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [0, 1, 1],
    ]);
    let Y = new Matrix([
      [-1, 0, 1, 2, 3, 4, 5],
      [0, 3, 5, 6, 79, 3, 1],
      [10, 11, 2, 3, 4, 7, 8],
      [1, 112, 0, 0, 0, 7, 8],
      [1000, 2, 56, 40, 1, 1, 3],
      [7, 6, 5, 4, 3, 2, 1],
    ]);
    let { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    let solution = new Matrix([
      [203.7567, 0, 0, 0, 0, 0, 0],
      [-149.1338, 3.3309, 2.0243, 1.5134, 0, 0, 0],
      [0, 0, 0, 0, 1.0827, 0.3911, 0.4738],
    ]);
    let result = Matrix.round(cssls(XtX, XtY, Pset, l, p).mul(10000)).mul(
      0.0001,
    );
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('Van Benthem - Keenan example', () => {
    let X = new Matrix([[95, 89, 82], [23, 76, 44], [61, 46, 62], [42, 2, 79]]);
    let Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);

    let { l, p, XtX, XtY, Pset } = initialisation(X, Y);
    let solution = new Matrix([
      [0, 0.6873, 0.2836],
      [0.6272, 0, 0.2862],
      [0.3517, 0.2873, 0.335],
    ]);
    let result = Matrix.round(cssls(XtX, XtY, Pset, l, p).mul(10000)).mul(
      0.0001,
    );
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('negative identity X, positive Y', () => {
    let X = Matrix.eye(3).mul(-1);
    let Y = new Matrix([[1], [2], [3]]);
    let init = initialisation(X, Y);
    let solution = new Matrix([[-1], [-2], [-3]]);
    let result = cssls(init.XtX, init.XtY, null, init.l, init.p);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
  it('test', () => {
    let X = Matrix.eye(3).mul(-1);
    let Y = new Matrix([[1], [2], [3]]);
    let init = initialisation(X, Y);
    let solution = new Matrix([[0], [0], [0]]);
    let result = cssls(
      init.XtX,
      init.XtY,
      selection(init.Pset, init.Fset),
      init.l,
      init.p,
    );
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
});
