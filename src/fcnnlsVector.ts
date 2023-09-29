import { Matrix } from 'ml-matrix';

import { KAndInfo, KOnly, FcnnlsOptions, fcnnls } from './fcnnls';

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
  options?: FcnnlsOptions<false | undefined>,
): KOnly;
export function fcnnlsVector(
  X: number[][] | Matrix,
  y: number[],
  options?: FcnnlsOptions<true>,
): KAndInfo;
export function fcnnlsVector<T extends boolean | undefined>(
  X: number[][] | Matrix,
  y: number[],
  options?: FcnnlsOptions<T>,
): KAndInfo | KOnly;
export function fcnnlsVector<T extends boolean | undefined>(
  X: number[][] | Matrix,
  y: number[],
  options: FcnnlsOptions<T> = {},
) {
  if (!Array.isArray(y)) {
    throw new TypeError('y must be a 1D Array');
  }
  const Y = Matrix.columnVector(y);
  return fcnnls(X, Y, options);
}
