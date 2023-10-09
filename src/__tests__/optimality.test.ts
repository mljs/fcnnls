import { Matrix } from 'ml-matrix';
import { expect, it, describe } from 'vitest';

import { initialisation } from '../initialisation';
import { optimality } from '../optimality';

import { prepareInput } from './prepareInitInput';

describe('optimality test', () => {
  it('identity X, Y 1-dimension', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const input = prepareInput(X, Y);
    const { XtX, XtY } = input;
    const { iter, W, K, Pset, Fset } = initialisation(input);
    const l = X.columns;
    const p = Y.columns;
    const D = K.clone();
    const result = optimality({
      iter,
      maxIter: X.columns * 3,
      XtX,
      XtY,
      Fset,
      Pset,
      W,
      K,
      l,
      p,
      D,
      gradientTolerance: 10e-10,
    });

    expect(result.Pset).toEqual([[1, 2, 3]]);
    expect(result.Fset).toEqual([]);
  });
});
