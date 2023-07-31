import { Matrix } from 'ml-matrix';

import fcnnls from './fcnnls';

export interface FcnnlsVectorOptions {
  maxIterations?: number;
}
/**
 * Fast Combinatorial Non-negative Least Squares with single Right Hand Side
 * @param X - input data matrix
 * @param y - output data vector
 * @param options - for maxIterations
 * @returns solution vector k
 */
export default function fcnnlsVector(
  X: Matrix,
  y: number[],
  options: FcnnlsVectorOptions = {},
) {
  if (!Array.isArray(y)) {
    throw new TypeError('y must be a 1D Array');
  }
  const Y = Matrix.columnVector(y);
  const K = fcnnls(X, Y, options);
  const k = K.to1DArray();
  return k;
}
