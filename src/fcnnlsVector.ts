import { Matrix } from 'ml-matrix';

import fcnnls, { type FcnnlsOptions } from './fcnnls';

/**
 * Fast Combinatorial Non-negative Least Squares with single Right Hand Side
 * @param X - input data matrix
 * @param y - output data vector
 * @param options - for maxIterations
 * @returns Solution vector.
 */
export default function fcnnlsVector(
  X: Matrix,
  y: number[],
  options: FcnnlsOptions = {},
) {
  if (!Array.isArray(y)) {
    throw new TypeError('y must be a 1D Array');
  }
  const Y = Matrix.columnVector(y);
  const K = fcnnls(X, Y, options);
  const k = K.to1DArray();
  return k;
}
