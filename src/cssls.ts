import {
  Matrix,
  LuDecomposition,
  solve,
  CholeskyDecomposition,
} from 'ml-matrix';

import { sortCollectionSet } from './util/sortCollectionSet';

/**
 * Combinatorial Subspace Least Squares - subfunction for the FC-NNLS
 * Solves XtX*K = XtY for the variables in Pset
 * if XtX (or XtX(vars,vars)) is singular, performs the svd and find pseudo-inverse, otherwise (even if ill-conditioned) finds inverse with LU decomposition and solves the set of equations
 * it is consistent with matlab results for ill-conditioned matrices (at least consistent with test 'ill-conditioned square X rank 2, Y 3x1' in cssls.test)
 * @param Cssls object, @see {@link Cssls}
 */
export function cssls({ XtX, XtY, Pset, nColsX, nColsY }: Cssls): Matrix {
  let K = Matrix.zeros(nColsX, nColsY);
  if (Pset === null) {
    // used for initialisation where OLS is solved.
    const choXtX = new CholeskyDecomposition(XtX);
    if (choXtX.isPositiveDefinite()) {
      K = choXtX.solve(XtY);
    } else {
      const luXtX = new LuDecomposition(XtX);
      if (!luXtX.isSingular()) {
        K = luXtX.solve(Matrix.eye(nColsX)).mmul(XtY);
      } else {
        K = solve(XtX, XtY, true);
      }
    }
  } else {
    const { values: sortedPset, indices: sortedEset } = sortCollectionSet(Pset);
    if (
      sortedPset.length === 1 &&
      sortedPset[0].length === 0 &&
      sortedEset[0].length === nColsY
    ) {
      return K;
    } else if (
      sortedPset.length === 1 &&
      sortedPset[0].length === nColsX &&
      sortedEset[0].length === nColsY
    ) {
      const choXtX = new CholeskyDecomposition(XtX);
      if (choXtX.isPositiveDefinite()) {
        K = choXtX.solve(XtY);
      } else {
        const luXtX = new LuDecomposition(XtX);
        if (!luXtX.isSingular()) {
          K = luXtX.solve(Matrix.eye(nColsX)).mmul(XtY);
        } else {
          K = solve(XtX, XtY, true);
        }
      }
    } else {
      for (let k = 0; k < sortedPset.length; k++) {
        const cols2Solve = sortedEset[k];
        const vars = sortedPset[k];
        let L;
        const choXtX = new CholeskyDecomposition(XtX.selection(vars, vars));
        if (choXtX.isPositiveDefinite()) {
          L = choXtX.solve(XtY.selection(vars, cols2Solve));
        } else {
          const luXtX = new LuDecomposition(XtX.selection(vars, vars));
          if (!luXtX.isSingular()) {
            L = luXtX
              .solve(Matrix.eye(vars.length))
              .mmul(XtY.selection(vars, cols2Solve));
          } else {
            L = solve(
              XtX.selection(vars, vars),
              XtY.selection(vars, cols2Solve),
              true,
            );
          }
        }
        for (let i = 0; i < L.rows; i++) {
          for (let j = 0; j < L.columns; j++) {
            K.set(vars[i], cols2Solve[j], L.get(i, j));
          }
        }
      }
    }
  }
  return K;
}

interface Cssls {
  /**
   * XtX - Gram matrix
   */
  XtX: Matrix;
  /**
   * XtY
   */
  XtY: Matrix;
  /**
   * Pset - Subset of matrix K with positive values (indices)
   */
  Pset: number[][] | null;
  /**
   * nColsX - number of columns of X
   */
  nColsX: number;
  /**
   * nColsY - number of columns of Y
   */
  nColsY: number;
}
