import { toBeDeepCloseTo, toMatchCloseTo } from 'jest-matcher-deep-close-to';
import { Matrix } from 'ml-matrix';

import fcnnls from '../fcnnls';

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

describe('myModule test', () => {
  it('identity X, Y 4x1', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = fcnnls(X, Y, false);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
});
