import { Matrix } from 'ml-matrix';

import fcnnls from '../fcnnls';

describe('myModule test', () => {
  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = fcnnls(X, Y);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.columns; j++) {
        expect(result.get(i, j)).toBeCloseTo(solution.get(i, j));
      }
    }
  });
});
