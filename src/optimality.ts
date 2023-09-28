import { type Matrix } from 'ml-matrix';

import { setDifference } from './util/setDifference';

/**
 * Checks whether the solution has converged
 * @param see {@link OptimalityParams} for a description.
 * @returns Pset, Fset, W
 */
export function optimality({
  iter,
  maxIter,
  XtX,
  XtY,
  Fset,
  Pset,
  W,
  K,
  l,
  p,
  D,
  gradientTolerance,
}: OptimalityParams) {
  if (iter === maxIter) {
    throw new Error(
      'Maximum number of iterations exceeded. You may try to gradually increase the option.gradientTolerance.',
    );
  }

  // Check solution for optimality
  const V = XtY.subMatrixColumn(Fset).subtract(
    XtX.mmul(K.subMatrixColumn(Fset)),
  );
  for (let j = 0; j < Fset.length; j++) {
    // for the "negative" columns, we set the new gradient.
    W.setColumn(Fset[j], V.subMatrixColumn([j]));
  }
  const Jset: number[] = [];
  const fullSet: number[] = [];
  for (let i = 0; i < l; i++) {
    fullSet.push(i);
  }
  for (const colIndex of Fset) {
    const notPset = setDifference(fullSet, Pset[colIndex]);
    if (notPset.length === 0) {
      Jset.push(colIndex);
    } else if (W.selection(notPset, [colIndex]).max() <= gradientTolerance) {
      Jset.push(colIndex);
    }
  }
  Fset = setDifference(Fset, Jset);

  // For non-optimal solutions, add the appropriate variables to Pset
  if (Fset.length !== 0) {
    for (let j = 0; j < Fset.length; j++) {
      for (let i = 0; i < l; i++) {
        if (Pset[Fset[j]].includes(i)) W.set(i, Fset[j], -Infinity);
      }
      Pset[Fset[j]].push(W.subMatrixColumn(Fset).maxColumnIndex(j)[0]);
    }
    for (const colIndex of Fset) {
      D.setColumn(colIndex, K.getColumn(colIndex));
    }
  }
  for (let j = 0; j < p; j++) {
    Pset[j].sort((a, b) => a - b);
  }
  return { Pset, Fset, W };
}

/**
 * @param iter - current iteration
 * @param maxIter - maximum number of iterations, @default 3 times the number of columns of X
 * @param XtX - Gram matrix
 * @param XtY
 * @param Fset - Columns to be optimized (active), it stores indices of columns with negative values
 * @param Pset - Subset of matrix K with positive values (indices)
 * @param W - Gradient Matrix
 * @param K - Coefficients Matrix
 * @param l - Number of columns of X
 * @param p - Number of columns of X
 * @param D - K clone
 * @param gradientTolerance - Control over the optimality of the solution; applied over the largest gradient value of all. @default 1e-5.
 */
interface OptimalityParams {
  iter: number;
  Pset: number[][];
  Fset: number[];
  W: Matrix;
  XtX: Matrix;
  XtY: Matrix;
  K: Matrix;
  D: Matrix;
  p: number;
  l: number;
  maxIter: number;
  gradientTolerance: number;
}
