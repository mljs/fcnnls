import { Matrix } from 'ml-matrix';

import initialisation from '../initialisation';

describe('initialisation test', () => {
  it('identity X, Y 1-dimension', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let result = initialisation(X, Y);
    let solution = [0];
    expect(result.Fset).toStrictEqual(solution);
  });
  it('Van Benthem - Keenan example', () => {
    let X = new Matrix([
      [95, 89, 82],
      [23, 76, 44],
      [61, 46, 62],
      [42, 2, 79],
    ]);
    let Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);
    let { Pset } = initialisation(X, Y);

    let solutionPset = [
      [1, 2],
      [0, 2],
      [0, 1, 2],
    ];

    expect(Pset).toStrictEqual(solutionPset);
  });
  it('non-singular square X, Y 3x1', () => {
    let X = new Matrix([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
    let Y = new Matrix([[-1], [2], [-3]]);
    let result = initialisation(X, Y);
    let solution = [[0, 2]];
    expect(result.Pset).toStrictEqual(solution);
  });
});
