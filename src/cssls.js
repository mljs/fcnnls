'use strict';

const { Matrix, inverse } = require('ml-matrix');

const sortCollectionSet = require('./util/sortCollectionSet');
/**
 * (Combinatorial Subspace Least Squares) - subfunction for the FC-NNLS
 * @param {Matrix} XtX
 * @param {Matrix} XtY
 * @param {Array} Pset
 * @param {Numbers} l
 * @param {Numbers} p
 */
function cssls(XtX, XtY, Pset, l, p) {
  // Solves the set of equation XtX*K = XtY for the variables in Pset
  let K;
  if (Pset === null) {
    //add some error case
    K = inverse(XtX).mmul(XtY);
  } else {
    K = Matrix.zeros(l, p);
    let sortedPset = sortCollectionSet(Pset).values;
    let sortedEset = sortCollectionSet(Pset).indices;
    for (let k = 0; k < sortedPset.length; k++) {
      let cols2Solve = sortedEset[k];
      let vars = sortedPset[k];
      let L = inverse(XtX.selection(vars, vars), { useSVD: true }).mmul(
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

module.exports = cssls;
