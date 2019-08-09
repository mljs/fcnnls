'use strict';

const { Matrix } = require('ml-matrix');

const fcnnls = require('./fcnnls');

/**
 * Fast Combinatorial Non-negative Least Squares with single Right Hand Side
 * @param {Matrix or 2D Array} X
 * @param {1D Array} y
 * @param {object} [options={}]
 * @param {boolean} [maxIterations] if true or empty maxIterations is set at 3 times the number of columns of X
 * @returns {Array} k
 */
function fcnnlsVector(X, y, options = {}) {
  if (Array.isArray(y) === false) {
    throw new TypeError('y must be a 1D Array');
  }
  let Y = Matrix.columnVector(y);
  let K = fcnnls(X, Y, options);
  let k = K.to1DArray();
  return k;
}

module.exports = fcnnlsVector;
