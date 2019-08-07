'use strict';

const { Matrix } = require('ml-matrix');

const cssls = require('./cssls');

function initialisation(X, Y) {
  // check if input error
  // error(nargchk(2,2....)) à traduire
  let n = X.rows;
  let l = X.columns;
  let p = Y.columns;
  let iter = 0;
  let maxiter = 3 * l;

  if (Y.rows !== n) return 'ERROR: matrix size not compatible'; // end function, ERROR to be handled

  let W = Matrix.zeros(l, p);

  // precomputes part of pseudoinverse
  let XtX = X.transpose().mmul(X);
  let XtY = X.transpose().mmul(Y);

  let K = cssls(XtX, XtY, null); // K is lxp
  let Pset = []; // A better way to find the Pset ? using a function like filter or equivalent ?
  for (let j = 0; j < p; j++) {
    Pset[j] = [];
    for (let i = 0; i < l; i++) {
      if (K.get(i, j) > 0) {
        Pset[j].push(i);
      } else {
        K.set(i, j, 0);
      } //This is our initial solution, it's the solution found by overwriting the unconstrained least square solution
    }
  }

  let Fset = [];
  for (let j = 0; j < p; j++) {
    if (Pset[j].length !== l) {
      Fset.push(j);
    }
  }

  let D = K.clone();

  return { n, l, p, iter, maxiter, W, XtX, XtY, K, Pset, Fset, D };
}

module.exports = initialisation;