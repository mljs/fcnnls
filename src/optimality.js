'use strict';

const { Matrix } = require('ml-matrix');

const setDifference = require('./util/setDifference');

// Make sure the solution has converged

function optimality(iter, maxiter, XtX, XtY, Fset, Pset, W, K, l, D) {
  if (iter === maxiter) return 'Maximum number of iterations exceeded';
  // Check solution for optimality
  let V = XtY.subMatrixColumn(Fset).subtract(XtX.mmul(K.subMatrixColumn(Fset)));

  for (let j = 0; j < Fset.length; j++) {
    W.setColumn(Fset[j], V.subMatrixColumn([j]));
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
      Pset[Fset[j]].push(W.subMatrixColumn(Fset).maxColumnIndex(j));
    }
    for (let j = 0; j < Fset.length; j++) {
      D.setColumn(Fset[j], K.getColumn(Fset[j]));
    }
  }
}

module.exports = optimality;
