'use strict';

const { Matrix, inverse } = require('ml-matrix');

const selection = require('/src/array-utils/selection');
/**
 *
 * @param {Matrix} XtX
 * @param {Matrix} XtY
 * @param {Array} Pset
 */
function cssls(XtX, XtY, Pset) {
  // Solves the set of equation XtX*K = XtY for the variables in Pset
  let K;
  if (Pset === null) {
    //add some error case
    K = inverse(XtX).mmul(XtY);
  } else {
    let l = XtY.rows; //XtY is lxp matrix
    let p = XtY.columns;
    K = Matrix.zeros(l, p);
    let v = new Array(l);
    for (let i = 0; i < l; i++) {
      v[i] = Math.pow(2, l - 1 - i);
    }
    let codedPset = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < Pset[j].length; i++) {
        codedPset[j] += v[Pset[j][i]];
      }
    } // multiplication v = [. . . . .]*Pset (Pset as logical matrix of non-negative elements of initial feasible solution)
    let sortedPset = sortArray(codedPset)[0];
    let sortedEset = sortArray(codedPset)[1];
    let breaks = diff(sortedPset);
    let w = []; // Here we don't know the size of w before, is there a better method to initialise w ? Filter returns the value not the index..!
    for (let j = 0; j < breaks.length; j++) {
      if (breaks[j] > 0) w.push(j);
    }
    let breakIdx = [-1].concat(w, p - 1); // Indexes start at -1 to be good with JavaScript
    for (let k = 0; k < breakIdx.length - 1; k++) {
      let breaksIdxVect = [];
      for (let i = breakIdx[k] + 1; i <= breakIdx[k + 1]; i++) {
        breaksIdxVect.push(i);
      } // As before, how to initialise a vector which we don't know the size a priori ?
      let cols2Solve = selection(sortedEset, breaksIdxVect);
      let vars = Pset[sortedEset[breakIdx[k] + 1]];
      let L = inverse(XtX.selection(vars, vars)).mmul(
        XtY.selection(vars, cols2Solve),
      );
      for (let i = 0; i < L.rows; i++) {
        for (let j = 0; j < L.columns; j++) {
          K.set(vars[i], cols2Solve[j], L.get(i, j));
        }
      }
    }
  }
  return K;
}
