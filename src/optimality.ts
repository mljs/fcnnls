import { type Matrix } from 'ml-matrix';

import setDifference from './util/setDifference';

type Optimality = (
  // current iterations
  iter: number,
  // maximum number of iterations
  maxIter: number,
  // Gram matrix
  XtX: Matrix,
  XtY: Matrix,
  /* Columns to be optimized (active), it stores indices of columns with
   * negative values
   */
  Fset: number[],
  /* Columns not to be optimized (passive)
   * Each subarray corresponds to col of K
   * And stores indices of positive values of that column
   */
  Pset: number[][],
  // Gradient Matrix
  W: Matrix,
  // Coefficients Matrix
  K: Matrix,
  // Number of rows of X
  l: number,
  // Number of columns of X
  p: number,
  // K clone
  D: Matrix,
  // Number of decimals to chop number
  gradientToleranceDecimals: number,
) => {
  Pset: number[][];
  Fset: number[];
  W: Matrix;
};

/**
 * Checks whether the solution has converged
 * @param iter - current iteration
 * @param maxIter - maximum number of iterations
 * @param XtX - Gram matrix
 * @param XtY
 * @param Fset - Columns to be optimized (active), it stores indices of columns with negative values
 * @param Pset - Subset of matrix K with positive values (indices)
 * @param W - Gradient Matrix
 */
export const optimality: Optimality = function optimality(
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
  gradientToleranceDecimals,
) {
  if (iter === maxIter) {
    throw new Error('Maximum number of iterations exceeded');
  }

  // Check solution for optimality
  const V = XtY.subMatrixColumn(Fset).subtract(
    XtX.mmul(K.subMatrixColumn(Fset)),
  );
  for (let j = 0; j < Fset.length; j++) {
    // for the "negative" columns, we set the new gradient.
    W.setColumn(Fset[j], V.subMatrixColumn([j]));
  }
  const Jset = [];
  const fullSet = [];
  for (let i = 0; i < l; i++) {
    fullSet.push(i);
  }
  for (const colIndex of Fset) {
    const notPset = setDifference(fullSet, Pset[colIndex]);
    if (notPset.length === 0) {
      Jset.push(colIndex);
    } else if (
      parseFloat(
        W.selection(notPset, [colIndex])
          .max()
          .toFixed(gradientToleranceDecimals),
      ) <= 0
    ) {
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
};
