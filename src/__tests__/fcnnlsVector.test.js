'use strict';

const { Matrix } = require('ml-matrix');
const {
  toBeDeepCloseTo,
  toMatchCloseTo,
} = require('jest-matcher-deep-close-to');

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

const fcnnlsVector = require('../fcnnlsVector');

describe('myModule test', () => {
  it('example documentation', () => {
    let X = new Matrix([[1, 1, 2], [10, 11, -9], [-1, 0, 0], [-5, 6, -7]]);
    let y = [-1, 11, 0, 1];
    let solution = [0.461, 0.5611, 0];
    let result = fcnnlsVector(X, y);
    expect(result).toBeDeepCloseTo(solution, 4);
  });
});
