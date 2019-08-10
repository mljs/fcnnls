'use strict';

const { Matrix } = require('ml-matrix');
const {
  toBeDeepCloseTo,
  toMatchCloseTo,
} = require('jest-matcher-deep-close-to');

const fcnnls = require('../fcnnls');

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

describe('myModule test', () => {
  it('identity X, Y 4x1', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = fcnnls(X, Y, false);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it.only('random big X, ramdome big Y', () => {
    let X = Matrix.rand(120, 20);
    let Y = Matrix.rand(120, 300);
    console.log(X);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = fcnnls(X, Y, false);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });
});
