'use strict';

const { Matrix } = require('ml-matrix');
const {
  toBeDeepCloseTo,
  toMatchCloseTo,
} = require('jest-matcher-deep-close-to');

const initialisation = require('../initialisation');
const optimality = require('../optimality');

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

describe('optimality test', () => {
  it('identity X, Y 1-dimension', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let { l, iter, maxiter, W, XtX, XtY, K, Pset, Fset, D } = initialisation(
      X,
      Y,
    );
    let result = optimality(iter, maxiter, XtX, XtY, Fset, Pset, W, K, l, D);

    expect(result.Pset).toMatchCloseTo([[1, 2, 3]], 4);
    expect(result.Fset).toMatchCloseTo([], 4);
  });
});
