import { Matrix } from 'ml-matrix';

import { cssls } from './cssls';

/**
 * Solves as std linear squares,
 * and overwrites the negative values of K with 0 as an initial guess for K,
 * It also precomputes part of the pseudoinverse used to solve Least Squares.
 * @param - X input data matrix
 * @param - Y output data matrix
 * @returns - initial values for the algorithm (including the solution K to least squares, overwriting of negative values with 0)
 */
export default function initialisation(X: Matrix, Y: Matrix) {
  // X = n x l
  // W = l x p, same as K
  // Y = n x p
  const n = X.rows;
  const l = X.columns;
  const p = Y.columns;
  const iter = 0;

  if (Y.rows !== n) throw new Error('ERROR: matrix size not compatible');

  const W = Matrix.zeros(l, p);

  // precomputes part of pseudoinverse
  const XtX = X.transpose().mmul(X);
  const XtY = X.transpose().mmul(Y);

  const K = cssls(XtX, XtY, null, l, p); // K is lxp
  /* Each subarray corresponds to col of K
   * And stores indices of positive values of that column
   */
  const Pset: number[][] = [];
  for (let j = 0; j < p; j++) {
    Pset[j] = [];
    for (let i = 0; i < l; i++) {
      if (K.get(i, j) > 0) {
        Pset[j].push(i); // [[1,2,3...,l],[1,2,4,7,..,l],[],..] p arrays, each length l or less
      } else {
        K.set(i, j, 0);
      } // This is our initial solution, it's the solution found by overwriting the unconstrained least square solution in K.
    }
  }
  const Fset: number[] = [];
  for (let j = 0; j < p; j++) {
    if (Pset[j].length !== l) {
      Fset.push(j); // If column j of K was not all positive, add it to the Fset. So Fset are the indices of columns with negative values
    }
  }

  const D = K.clone();

  return { n, l, p, iter, W, XtX, XtY, K, Pset, Fset, D };
}
