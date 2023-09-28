import { Matrix } from 'ml-matrix';

import { cssls } from './cssls';

interface Initialisation {
  XtX: Matrix;
  XtY: Matrix;
  nRowsX: number;
  nColsX: number;
  nRowsY: number;
  nColsY: number;
}
/**
 * Solves OLS problem,  overwriting the negative values of K with 0.
 * It also pre-computes part of the pseudo-inverse used to solve Least Squares.
 * @param XtX - input data matrix
 * @param XtY - output data matrix
 * @returns initial values for the algorithm (including the solution K to least squares, overwriting of negative values with 0)
 */
export function initialisation({
  XtX,
  XtY,
  nRowsX,
  nColsX,
  nRowsY,
  nColsY,
}: Initialisation) {
  const iter = 0;

  if (nRowsY !== nRowsX) throw new Error('ERROR: matrix size not compatible');

  const W = Matrix.zeros(nColsX, nColsY);
  const K = cssls({ XtX, XtY, Pset: null, nColsX, nColsY }); //K same dim as W
  /*
   * Each subarray corresponds to col of K
   * And stores indices of positive values of that column
   */
  const Pset: number[][] = [];
  for (let j = 0; j < nColsY; j++) {
    Pset[j] = []; // An array of indices/column. These are of positive values.
    for (let i = 0; i < nColsX; i++) {
      if (K.get(i, j) > 0) {
        Pset[j].push(i);
      } else {
        K.set(i, j, 0);
      } // Initial solution K, overwriting OLS solution.
    }
  }
  const Fset: number[] = [];
  for (let j = 0; j < nColsY; j++) {
    if (Pset[j].length !== nColsX) {
      Fset.push(j); // If column j of K was not all positive, add it to the Fset. So Fset are the indices of columns with negative values
    }
  }

  return { iter, W, K, Pset, Fset };
}
