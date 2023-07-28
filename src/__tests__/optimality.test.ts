import { Matrix } from 'ml-matrix';

import initialisation from '../initialisation';
import { optimality } from '../optimality';

describe('optimality test', () => {
  it('identity X, Y 1-dimension', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const { l, iter, W, XtX, XtY, K, Pset, Fset, D, p } = initialisation(X, Y);
    const result = optimality(
      iter,
      X.columns * 3,
      XtX,
      XtY,
      Fset,
      Pset,
      W,
      K,
      l,
      p,
      D,
      10,
    );

    expect(result.Pset).toEqual([[1, 2, 3]]);
    expect(result.Fset).toEqual([]);
  });
});
