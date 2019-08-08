'use strict';

const { Matrix } = require('ml-matrix');

const fcnnls = require('./fcnnls');

function fcnnlsVector(X, y, options = {}) {
  if (Array.isArray(y) === false) {
    return 'ERROR: y must be an array';
  }
  let Y = Matrix.columnVector(y);
  let K = fcnnls(X, Y, options);
  let k = K.to1DArray;

  return k;
}

module.exports = fcnnlsVector;
