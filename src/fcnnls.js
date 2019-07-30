'use strict';

// changer en typed array
// Would it be sensible to be able to extract a full row(s)/column(s) easily with matrix.selection ?
// add errors...

const { Matrix, inverse } = require('ml-matrix');

const selection = require('./array-utils/selection');

module.exports = fcnnls;

function fcnnls(X, Y) {
  // check if input error
  // error(nargchk(2,2....)) à traduire
  let n = X.rows;
  let l = X.columns;
  let p = Y.columns;
  let iter = 0;
  let maxiter = 3 * l;

  if (Y.rows !== n) return 'ERROR: matrix size not compatible'; // end function

  let W = Matrix.zeros(l, p);

  // precomputes part of pseudoinverse
  let XtX = X.transpose().mmul(X);
  let XtY = X.transpose().mmul(Y);

  // Obtain the initial feasible solition and corresponding passive set
  let K = cssls(XtX, XtY, null); // K is lxp
  let Pset = new Array(p).fill([]);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < l; i++) {
      if (K.get(i, j) > 0) {
        if (Pset[j].length === 0) {
          Pset[j][0] = i;
        } else {
          Pset[j].push(i);
        }
      } else {
        K.set(i, j, 0);
      } //If non-positive we set it to zero
    }
  } // A better way to find the Pset ? using a function like filter or equivalent ?
  let D = Object.assign({}, K); // not sure but here we probably want to copy K without changing K when we modify D.
  let Fset = [];
  for (let j = 0; j < p; j++) {
    if (Pset[j].length !== l) {
      if (Fset.length === 0) {
        Fset[0] = j;
      } else {
        Fset.push(j);
      }
    }
  }

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

function setDifference(a, b) {
  let c = [];
  for (let i of a) {
    if (!b.includes(i)) c.push(i);
  }
  return c;
}

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
        if (breaksIdxVect.length === 0) {
          breaksIdxVect[0] = i;
        } else {
          breaksIdxVect.push(i);
        }
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

function sortArray(v) {
  v.sort((a, b) => {
    if (a.value === b.value) return a.index - b.index;
    return a.value - b.value;
  });

  let values = v.map((item) => item.value);
  let indices = v.map((item) => item.index);
  return [values, indices];
}

function diff(v) {
  let u = [];
  for (let i = 0; i < v.length - 1; i++) {
    u.push(v[i + 1] - v[i]);
  }
  return u;
}
