'use strict';

const { Matrix, inverse } = require('ml-matrix');

const sortCollectionSet = require('./util/sortCollectionSet');
/**
 * (Combinatorial Subspace Least Squares) - subfunction for the FC-NNLS
 * @private
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
    K = inverse(XtX).mmul(XtY);
  } else {
    let sortedPset = sortCollectionSet(Pset).values;
    let sortedEset = sortCollectionSet(Pset).indices;
    K = Matrix.zeros(l, p);
    if (
      sortedPset.length === 1 &&
      sortedPset[0].length === 0 &&
      sortedEset[0].length === p
    ) {
      return K;
    } else if (
      sortedPset.length === 1 &&
      sortedPset[0].length === l &&
      sortedEset[0].length === p
    ) {
      K = inverse(XtX).mmul(XtY);
    } else {
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
  }
  return K;
}

module.exports = cssls;
