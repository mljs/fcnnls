import { Matrix } from 'ml-matrix';

import { cssls } from './cssls';
import { initialisation } from './initialisation';
import { optimality } from './optimality';
import { selection, getRSE } from './util';

export interface FcnnlsOptions<T extends boolean | undefined> {
  /**
   * Number of iterations
   * @default 3 times the number of columns of X
   */
  maxIterations?: number;
  /**
   * Larger values (like 1e-4) could help if the number of iterations is exceeded. For most cases, lower values should also be fine.
   * @default 1e-5
   */
  gradientTolerance?: number;
  /**
   * Output the root squared error for each column of Y a well as the matrix K.
   * @default false
   */
  info?: T;
  /**
   * @default true. (The actual value is undefined.) `false` will add a column of ones to the left of X.
   */
  interceptAtZero?: boolean;
}

/**
 * Fast Combinatorial Non-negative Least Squares with multiple Right Hand Side
 * @param X - The data/input/predictors matrix
 * @param Y - The response matrix
 * @param options {@link FcnnlsOptions}
 * @returns By default, the object with the matrix of coefficients K. Please see {@link FcnnlsOutput} for more information.
 */
export function fcnnls(
  X: Matrix | number[][],
  Y: Matrix | number[] | number[][],
  options?: FcnnlsOptions<false | undefined>,
): KOnly;
export function fcnnls(
  X: Matrix | number[][],
  Y: Matrix | number[] | number[][],
  options?: FcnnlsOptions<true>,
): KAndInfo;
export function fcnnls<T extends boolean | undefined>(
  X: Matrix | number[][],
  Y: Matrix | number[] | number[][],
  options?: FcnnlsOptions<T>,
): KAndInfo | KOnly;
export function fcnnls<T extends boolean | undefined>(
  X: Matrix | number[][],
  Y: Matrix | number[] | number[][],
  options: FcnnlsOptions<T> = {},
) {
  X = Matrix.checkMatrix(X);
  Y = Matrix.checkMatrix(Y);

  // only in the case they explicitly set it false.
  if (options.interceptAtZero === false) {
    X = Matrix.ones(X.rows, X.columns + 1).setSubMatrix(X, 0, 1);
  }
  const {
    maxIterations = X.columns * 3,
    gradientTolerance = 1e-5,
    info = false,
  } = options;

  // pre-computes part of pseudo-inverse
  const Xt = X.transpose();
  const XtX = Xt.mmul(X);
  const XtY = Xt.mmul(Y);

  const { columns: nColsY, rows: nRowsY } = Y;
  const { columns: nColsX, rows: nRowsX } = X;

  const init = initialisation({ XtX, XtY, nRowsX, nColsX, nRowsY, nColsY });
  let { iter, W, Pset, Fset } = init;
  const K = init.K;
  const D = K.clone();

  // first RSE is the result of overwriting OLS result in K.
  const error = getRSE({ X, K, Y, error: new Matrix(1, nColsY) });
  const rse = [error.to1DArray()];

  // Active set algorithm for NNLS main loop
  while (Fset.length > 0) {
    // Solves for the passive variables (uses subroutine below)
    let L = cssls({
      XtX,
      XtY: XtY.subMatrixColumn(Fset),
      Pset: selection(Pset, Fset),
      nColsX,
      nColsY: Fset.length,
    });
    for (let i = 0; i < nColsX; i++) {
      for (let j = 0; j < Fset.length; j++) {
        K.set(i, Fset[j], L.get(i, j));
      }
    }

    // Finds any infeasible solutions
    const infeasIndex: number[] = [];
    for (let j = 0; j < Fset.length; j++) {
      for (let i = 0; i < nColsX; i++) {
        if (L.get(i, j) < 0) {
          infeasIndex.push(j);
          break;
        }
      }
    }
    let Hset = selection(Fset, infeasIndex);

    // Makes infeasible solutions feasible (standard NNLS inner loop)
    if (Hset.length > 0) {
      let m = Hset.length;
      const alpha = Matrix.ones(nColsX, m);

      while (m > 0 && iter < maxIterations) {
        iter++;

        alpha.mul(Infinity);

        // Finds indices of negative variables in passive set
        const hRowColIdx: [number[], number[]] = [[], []]; // Indexes work in pairs, each pair reprensents a single element, first array is row index, second array is column index
        const negRowColIdx: [number[], number[]] = [[], []]; // Same as before
        for (let j = 0; j < m; j++) {
          for (const item of Pset[Hset[j]]) {
            if (K.get(item, Hset[j]) < 0) {
              hRowColIdx[0].push(item); // i
              hRowColIdx[1].push(j);
              negRowColIdx[0].push(item); // i
              negRowColIdx[1].push(Hset[j]);
            } // Compared to matlab, here we keep the row/column indexing (we are not taking the linear indexing)
          }
        }

        for (let k = 0; k < hRowColIdx[0].length; k++) {
          // could be hRowColIdx[1].length as well
          alpha.set(
            hRowColIdx[0][k],
            hRowColIdx[1][k],
            D.get(negRowColIdx[0][k], negRowColIdx[1][k]) /
              (D.get(negRowColIdx[0][k], negRowColIdx[1][k]) -
                K.get(negRowColIdx[0][k], negRowColIdx[1][k])),
          );
        }

        let alphaMin: number[] | Matrix = [];
        const minIdx: number[] = [];
        for (let j = 0; j < m; j++) {
          alphaMin[j] = alpha.minColumn(j);
          minIdx[j] = alpha.minColumnIndex(j)[0];
        }

        alphaMin = Matrix.rowVector(alphaMin);
        for (let i = 0; i < nColsX; i++) {
          alpha.setSubMatrix(alphaMin, i, 0);
        }

        let E = new Matrix(nColsX, m);
        E = D.subMatrixColumn(Hset).subtract(
          alpha
            .subMatrix(0, nColsX - 1, 0, m - 1)
            .mul(D.subMatrixColumn(Hset).subtract(K.subMatrixColumn(Hset))),
        );
        for (let j = 0; j < m; j++) {
          D.setColumn(Hset[j], E.subMatrixColumn([j]));
        }

        const idx2zero = [minIdx, Hset];
        for (let k = 0; k < m; k++) {
          D.set(idx2zero[0][k], idx2zero[1][k], 0);
        }

        for (let j = 0; j < m; j++) {
          Pset[Hset[j]].splice(
            Pset[Hset[j]].findIndex((item) => item === minIdx[j]),
            1,
          );
        }

        L = cssls({
          XtX,
          XtY: XtY.subMatrixColumn(Hset),
          Pset: selection(Pset, Hset),
          nColsX,
          nColsY: m,
        });
        for (let j = 0; j < m; j++) {
          K.setColumn(Hset[j], L.subMatrixColumn([j]));
        }

        Hset = [];
        for (let j = 0; j < K.columns; j++) {
          for (let i = 0; i < nColsX; i++) {
            if (K.get(i, j) < 0) {
              Hset.push(j);

              break;
            }
          }
        }
        m = Hset.length;

        if (info) {
          rse.push(getRSE({ X, K, Y, error }).to1DArray());
        }
      }
    }
    if (Hset.length === 0 || (iter === maxIterations && info)) {
      rse.push(getRSE({ X, K, Y, error }).to1DArray());
    }

    const newParam = optimality({
      iter,
      maxIter: maxIterations,
      XtX,
      XtY,
      Fset,
      Pset,
      W,
      K,
      l: nColsX,
      p: nColsY,
      D,
      gradientTolerance,
    });
    Pset = newParam.Pset;
    Fset = newParam.Fset;
    W = newParam.W;
  }
  if (info) {
    return { K, info: { iterations: rse.length, rse } };
  }

  return { K };
}

export interface Info {
  /**
   * Root Squared Error.
   * This is a row vector, the RSE values for each column of Y.
   */
  rse: number[][];
  /**
   * The number of times K was calculated (it accounts for the OLS guess of `K`, and will be `maxIterations + 1` when maxIterations is reached)
   */
  iterations: number;
}
export type FcnnlsOutput = KAndInfo | KOnly;
export interface KAndInfo {
  K: Matrix;
  info: Info;
}
export interface KOnly {
  K: Matrix;
}
