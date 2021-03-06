import { Matrix } from 'ml-matrix';

import fcnnls from './fcnnls';

/**
 * Fast Combinatorial Non-negative Least Squares with single Right Hand Side
 * @param {Matrix|number[][]} X
 * @param {number[]} y
 * @param {object} [options={}]
 * @param {boolean} [maxIterations] if true or empty maxIterations is set at 3 times the number of columns of X
 * @returns {Array} k
 */
export default function fcnnlsVector(X, y, options = {}) {
  if (Array.isArray(y) === false) {
    throw new TypeError('y must be a 1D Array');
  }
  let Y = Matrix.columnVector(y);
  let K = fcnnls(X, Y, options);
  let k = K.to1DArray();
  return k;
}
