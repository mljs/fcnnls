'use strict';

const { Matrix } = require('ml-matrix');

const initialisation = require('../initialisation');
const optimality = require('../optimality');
const cssls = require('../cssls');
const selection = require('../util/selection');

describe('optimality test', () => {
  it('identity X, Y 1-dimension', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let {
      n,
      l,
      p,
      iter,
      maxiter,
      W,
      XtX,
      XtY,
      K,
      Pset,
      Fset,
      D,
    } = initialisation(X, Y);
    let result = optimality(iter, maxiter, XtX, XtY, Fset, Pset, W, K, l, D);
    let solution = { Pset: [[1, 2, 3]], Fset: [], W: Matrix.zeros(4, 1) };
    expect(result).toStrictEqual(solution);
  });
  it('Van Benthem - Keenan example', () => {
    let X = new Matrix([[95, 89, 82], [23, 76, 44], [61, 46, 62], [42, 2, 79]]);
    let Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);
    let {
      n,
      l,
      p,
      iter,
      maxiter,
      W,
      XtX,
      XtY,
      K,
      Pset,
      Fset,
      D,
    } = initialisation(X, Y);
    let L = cssls(
      XtX,
      XtY.subMatrixColumn(Fset),
      selection(Pset, Fset),
      l,
      Fset.length,
    );
    for (let i = 0; i < l; i++) {
      for (let j = 0; j < Fset.length; j++) {
        K.set(i, Fset[j], L.get(i, j));
      }
    }
    let result = optimality(iter, maxiter, XtX, XtY, Fset, Pset, W, K, l, D);
    let solution = {
      Pset: [[1, 2], [0, 2], [0, 1, 2]],
      Fset: [],
      W: new Matrix([
        [-542.6068, -0.0, 0],
        [-0.0, -658.0763, 0],
        [-0.0, -0.0, 0],
      ]),
    };
    expect(result).toStrictEqual(solution);
  });
});
