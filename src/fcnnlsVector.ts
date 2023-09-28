import { Matrix } from 'ml-matrix';

import { type FcnnlsOptions, fcnnls } from './fcnnls';

/**
 * Fast Combinatorial Non-negative Least Squares with single Right Hand Side
 * @param X - input data matrix
 * @param y - output data vector
 * @param options - for maxIterations
 * @returns Solution vector.
 */
export function fcnnlsVector(
  X: number[][] | Matrix,
  y: number[],
  options: FcnnlsOptions = {},
) {
  if (!Array.isArray(y)) {
    throw new TypeError('y must be a 1D Array');
  }
  const Y = Matrix.columnVector(y);
  return fcnnls(X, Y, options);
}
