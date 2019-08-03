'use strict';

// changer en typed array
// Would it be sensible to be able to extract a full row(s)/column(s) easily with matrix.selection ?
// add errors...

const { Matrix } = require('ml-matrix');

const selection = require('./util/selection');
const setDifference = require('./util/setDifference');
const cssls = require('./cssls');
const initialisation = require('./initialisation');

module.exports = fcnnls;

function fcnnls(X, Y) {
  let {
    n,
    l,
    p,
    iter,
    maxiter,
    W,
    XtX,
    XtY,
    K,
    Pset,
    Fset,
    D,
  } = initialisation(X, Y);

  // Active set algortihm for NNLS main loop
  while (Fset.length > 0) {
    // Solves for the passive variables (uses subroutine below)
    let L = cssls(XtX, XtY.subMatrixColumn(Fset), selection(Pset, Fset));
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
          if (u.length === 0) {
            u[0] = j;
            break;
          } else {
            u.push(j);
            break;
          }
        }
      }
    }
    let Hset = selection(Fset, u);
    // Make infeasible solutions feasible (standard NNLS inner loop)
    if (Hset.length > 0) {
      let m = Hset.length;
      let alpha = Matrix.zeros(l, m);
      while (Hset.length > 0 || iter < maxiter) {
        iter++;
        for (let j = 0; j < m; j++) {
          for (let i = 0; i < l; i++) {
            alpha.set(i, j, Infinity);
          }
        }
        // Find indices of negative variables in passive set
        let hRowColIdx = [[], []]; // Indexes work in pairs, each of them reprensents a single element
        let negRowColIdx = [[], []]; //same as before
        for (let j = 0; j < m; j++) {
          for (let i = 0; i < l; i++) {
            if (Pset[Hset[j]][i] < 0 && K.get(i, Hset[j])) {
              if (hRowColIdx[0].length === 0) {
                hRowColIdx[0][0] = i;
                negRowColIdx[0][0] = i;
              } else {
                hRowColIdx[0].push(i);
                negRowColIdx[0].push(i);
              }
              if (hRowColIdx[1].length === 0) {
                hRowColIdx[1][0] = j;
                negRowColIdx[1][0] = Hset[j];
              } else {
                hRowColIdx[1].push(j);
                negRowColIdx[1].push(Hset[j]);
              }
            }
          }
        }
        for (let k = 0; k < hRowColIdx[0].length; k++) {
          // could be hRowColIdx[1].length as well
          alpha.set(
            hRowColIdx[0][k],
            hRowColIdx[1][k],
            D.get(hRowColIdx[0][k], hRowColIdx[1][k]) /
              (D.get(negRowColIdx[0][k], negRowColIdx[1][k]) -
                D.get(negRowColIdx[0][k], negRowColIdx[1][k])),
          );
        }
        let alphaMin = Float64Array(m);
        let minIdx = Float64Array(m);
        for (let j = 0; j < m; j++) {
          alphaMin[j] = alpha.minColumn(j);
          minIdx[j] = alpha.minColumnIndex(j);
        }
        for (let i = 0; i < l; i++) {
          alpha.setRow(i, alphaMin);
        }
        let E = D.subMatrixColumn(Hset, 0, -1);
        E = E - alpha.mul(E - K.subMatrixColumn(Hset, 0, -1));
        for (let j = 0; j < m; j++) {
          D.setColumn(j, E.subMatrixColumn(j, 0, -1));
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
        }
        let L = cssls(
          XtX,
          XtY.subMatrixColumn(Hset, 0, -1),
          selection(Pset, Hset),
        );
        for (let j = 0; j < m; j++) {
          K.setColumn(Hset[j], L.subMatrixColumn(j, 0, -1));
        }
        u = [];
        for (let j = 0; j < Fset.length; j++) {
          for (let i = 0; i < l; i++) {
            if (L.get(i, j) < 0) {
              if (u.length === 0) {
                u[0] = j;
                break;
              } else {
                u.push(j);
                break;
              }
            }
          }
        }
        Hset = selection(Fset, u);
      }
    }

    // Make sure the solution has converged
    if (iter === maxiter) return 'Maximum number of iterations exceeded';
    // Check solution for optimality
    let V =
      XtY.subMatrixColumn(Fset, 0, -1) -
      XtX.mmul(K.subMatrixColumn(Fset, 0, -1));
    for (let j = 0; j < Fset.length; j++) {
      W.setColumn(Fset[j], V.subMatrixColumn(j, 0, -1));
    }
    let Jset = [];
    for (let j = 0; j < Fset.length; j++) {
      if (Pset[Fset[j]].length === 0 && W.maxColumn(Fset[j]) <= 0) {
        if (Jset.length === 0) {
          Jset[0] = Fset[j];
        } else {
          Jset.push(Fset[j]);
        }
      }
    }
    Fset = setDifference(Fset, Jset);
    // For non-optimal solutions, add the appropriate variable to Pset
    if (Fset.length !== 0) {
      for (let j = 0; j < Fset.length; j++) {
        for (let i = 0; i < l; i++) {
          if (Pset[Fset[j]].includes(i)) W.set(i, Fset[j], -Infinity);
        }
        Pset[Fset[j]].push(W.subMatrixColumn(Fset, 0, -1).maxColumnIndex(j));
      }
      for (let j = 0; j < Fset.length; j++) {
        D.setColumn(Fset[j], K.getColumn(Fset[j]));
      }
    }
  }
}
