import { Matrix } from 'ml-matrix';

import { cssls } from './cssls';
import { initialisation } from './initialisation';
import { optimality } from './optimality';
import selection from './util/selection';

export interface FcnnlsOptions {
  /**
   * Number of iterations
   * @default 3 times the number of columns of X
   */
  maxIterations?: number;
  /**
   * Control over the optimality of the solution; applied over the largest gradient value of all.
   * @default 1e-5
   * Smaller values are less tolerant.
   */
  gradientTolerance?: number;
}

/**
 * Fast Combinatorial Non-negative Least Squares with multiple Right Hand Side
 * @param X
 * @param Y
 * @param options - see all the options here {@link FcnnlsOptions}
 * @returns Solution Matrix.
 */

export default function fcnnls(
  X: Matrix | number[][],
  Y: Matrix | number[] | number[][],
  options: FcnnlsOptions = {},
) {
  X = Matrix.checkMatrix(X);
  Y = Matrix.checkMatrix(Y);
  const init = initialisation(X, Y);
  const { l, p, XtX, XtY, K, D } = init;
  let { iter, W, Pset, Fset } = init;

  const { maxIterations = X.columns * 3, gradientTolerance = 1e-5 } = options;

  // Active set algorithm for NNLS main loop
  while (Fset.length > 0) {
    // Solves for the passive variables (uses subroutine below)
    let L = cssls(
      XtX,
      XtY.subMatrixColumn(Fset),
      selection(Pset, Fset),
      l,
      Fset.length,
    );
    for (let i = 0; i < l; i++) {
      for (let j = 0; j < Fset.length; j++) {
        K.set(i, Fset[j], L.get(i, j));
      }
    }

    // Finds any infeasible solutions
    const infeasIndex: number[] = [];
    for (let j = 0; j < Fset.length; j++) {
      for (let i = 0; i < l; i++) {
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
      const alpha = Matrix.ones(l, m);

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
        for (let i = 0; i < l; i++) {
          alpha.setSubMatrix(alphaMin, i, 0);
        }

        let E = new Matrix(l, m);
        E = D.subMatrixColumn(Hset).subtract(
          alpha
            .subMatrix(0, l - 1, 0, m - 1)
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

        L = cssls(XtX, XtY.subMatrixColumn(Hset), selection(Pset, Hset), l, m);
        for (let j = 0; j < m; j++) {
          K.setColumn(Hset[j], L.subMatrixColumn([j]));
        }

        Hset = [];
        for (let j = 0; j < K.columns; j++) {
          for (let i = 0; i < l; i++) {
            if (K.get(i, j) < 0) {
              Hset.push(j);

              break;
            }
          }
        }
        m = Hset.length;
      }
    }

    const newParam = optimality(
      iter,
      maxIterations,
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
    );
    Pset = newParam.Pset;
    Fset = newParam.Fset;
    W = newParam.W;
  }

  return K;
}
