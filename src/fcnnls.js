'use strict';

// add errors...

const { Matrix } = require('ml-matrix');

const selection = require('./util/selection');
// const sortCollectionSet = require('./util/sortCollectionSet');
const cssls = require('./cssls');
const initialisation = require('./initialisation');
const optimality = require('./optimality');

/**
 * Fast Combinatorial Non-negative Least Squares with multiple Right Hand Side
 * @param {Matrix or 2D Array} X
 * @param {Matrix} Y
 * @param {object} [options={}]
 * @param {boolean} [maxIterations] if true maxIterations is set at 3 times the number of columns of X
 * @returns {Matrix} K
 */

function fcnnls(X, Y, options = {}) {
  X = Matrix.checkMatrix(X);
  Y = Matrix.checkMatrix(Y);
  let { l, p, iter, W, XtX, XtY, K, Pset, Fset, D } = initialisation(X, Y);
  const { maxIterations = X.columns * 3 } = options;

  // Active set algortihm for NNLS main loop
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
    // find any infeasible solutions
    let u = [];
    for (let j = 0; j < Fset.length; j++) {
      for (let i = 0; i < l; i++) {
        if (L.get(i, j) < 0) {
          u.push(j);
          break;
        }
      }
    }
    let Hset = selection(Fset, u);
    // Make infeasible solutions feasible (standard NNLS inner loop)
    if (Hset.length > 0) {
      let m = Hset.length;
      let alpha = Matrix.zeros(l, m);
      while (Hset.length > 0 && iter < maxIterations) {
        iter++;
        for (let j = 0; j < m; j++) {
          for (let i = 0; i < l; i++) {
            alpha.set(i, j, Infinity);
          }
        }
        // Find indices of negative variables in passive set
        let hRowColIdx = [[], []]; // Indexes work in pairs, each of them reprensents a single element, first array is row index, second array is column index
        let negRowColIdx = [[], []]; //same as before
        for (let j = 0; j < m; j++) {
          for (let i = 0; i < Pset[Hset[j]].length; i++) {
            if (K.get(Pset[Hset[j]][i], Hset[j]) < 0) {
              hRowColIdx[0].push(i);
              negRowColIdx[0].push(i);
              hRowColIdx[1].push(j);
              negRowColIdx[1].push(Hset[j]);
            }
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
        // console.log(alpha), pas tout à fait la même première composante, erreur numérique ? très proche de zéro !
        let alphaMin = [];
        let minIdx = [];
        for (let j = 0; j < m; j++) {
          alphaMin[j] = alpha.minColumn(j);
          minIdx[j] = alpha.minColumnIndex(j)[0];
        }
        for (let i = 0; i < l; i++) {
          alpha.setRow(i, alphaMin);
        }
        let E = D.subMatrixColumn(Hset);
        E = D.subMatrixColumn(Hset).subtract(
          alpha.mul(D.subMatrixColumn(Hset).subtract(K.subMatrixColumn(Hset))),
        );
        for (let j = 0; j < m; j++) {
          D.setColumn(j, E.subMatrixColumn([j]));
        }
        let idx2zero = [minIdx, Hset];
        for (let k = 0; k < m; k++) {
          D.set(idx2zero[0][k], idx2zero[1][k], 0);
        }

        for (let j = 0; j < m; j++) {
          Pset[Hset[j]].splice(
            Pset[Hset[j]].findIndex((item) => item === minIdx[j]),
            1,
          );
        } // à retester avec exemple plus complexe

        L = cssls(XtX, XtY.subMatrixColumn(Hset), selection(Pset, Hset), l, m);
        for (let j = 0; j < m; j++) {
          K.setColumn(Hset[j], L.subMatrixColumn([j]));
        }
        u = [];
        for (let j = 0; j < Fset.length; j++) {
          for (let i = 0; i < l; i++) {
            if (L.get(i, j) < 0) {
              u.push(j);
              break;
            }
          }
        }
        Hset = selection(Fset, u);
      }
    }

    let newParam = optimality(
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
    );
    Pset = newParam.Pset;
    Fset = newParam.Fset;
    W = newParam.W;
  }
  return K;
}

module.exports = fcnnls;
