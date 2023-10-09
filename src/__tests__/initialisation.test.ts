import { Matrix } from 'ml-matrix';
import { expect, it, describe } from 'vitest';

import { initialisation } from '../initialisation';

import { prepareInput } from './prepareInitInput';

describe('initialisation test', () => {
  it('identity X, Y 1-dimension', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const result = initialisation(prepareInput(X, Y));
    const solution = [0];
    expect(result.Fset).toStrictEqual(solution);
  });
  it('Van Benthem - Keenan example', () => {
    const X = new Matrix([
      [95, 89, 82],
      [23, 76, 44],
      [61, 46, 62],
      [42, 2, 79],
    ]);
    const Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);
    const { Pset } = initialisation(prepareInput(X, Y));

    const solutionPset = [
      [1, 2],
      [0, 2],
      [0, 1, 2],
    ];

    expect(Pset).toStrictEqual(solutionPset);
  });
  it('non-singular square X, Y 3x1', () => {
    const X = new Matrix([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
    const Y = new Matrix([[-1], [2], [-3]]);
    const result = initialisation(prepareInput(X, Y));
    const solution = [[0, 2]];
    expect(result.Pset).toStrictEqual(solution);
  });
});
