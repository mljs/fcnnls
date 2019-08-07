'use strict';

const setDifference = require('./util/setDifference');

// Make sure the solution has converged

function optimality(iter, maxiter, XtX, XtY, Fset, Pset, W, K, l, p, D) {
  if (iter === maxiter) return 'Maximum number of iterations exceeded';
  // Check solution for optimality
  let V = XtY.subMatrixColumn(Fset).subtract(XtX.mmul(K.subMatrixColumn(Fset)));
  for (let j = 0; j < Fset.length; j++) {
    W.setColumn(Fset[j], V.subMatrixColumn([j]));
  }
  let Jset = [];
  let fullSet = [];
  for (let i = 0; i < l; i++) {
    fullSet.push(i);
  }
  for (let j = 0; j < Fset.length; j++) {
    let notPset = setDifference(fullSet, Pset[Fset[j]]);
    if (W.selection(notPset, [Fset[j]]).max() <= 0) {
      Jset.push(Fset[j]);
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
    for (let j = 0; j < Fset.length; j++) {
      D.setColumn(Fset[j], K.getColumn(Fset[j]));
    }
  }
  for (let j = 0; j < p; j++) {
    Pset[j].sort();
  }
  return { Pset, Fset, W };
}

module.exports = optimality;
