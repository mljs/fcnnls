'use strict';

const { Matrix } = require('ml-matrix');

const fcnnls = require('../index.js');

const X = new Matrix([[95, 89, 82], [23, 76, 44], [61, 46, 62], [42, 2, 79]]);
const Y = new Matrix([[92, 99, 80], [74, 19, 43], [18, 41, 51], [41, 61, 39]]);
const K = new Matrix([
  [0, 0.6872687, 0.2835705],
  [0.6272475, 0, 0.2861623],
  [0.3516573, 0.2873328, 0.334968],
]);

describe('myModule test', () => {
  it('simple', () => {
    expect(fcnnls(X, Y)).toStrictEqual(K);
  });
});
