import { Matrix } from 'ml-matrix';
import { expect } from 'vitest';

import { type FcnnlsOutput } from '../fcnnls';

// used for most tests here and in fcnnlsVector.test.ts
export function assertResult(
  result: FcnnlsOutput,
  solution: Matrix,
  precision = 4,
) {
  result = result instanceof Matrix ? result : result.K;
  solution = Matrix.checkMatrix(solution);
  for (let i = 0; i < result.rows; i++) {
    for (let j = 0; j < result.columns; j++) {
      const sol = solution.get(i, j);
      // for numbers > 1000 just match up to the decimal point.
      if (sol > 10e2) precision = 0;
      expect(result.get(i, j)).toBeCloseTo(sol, precision);
    }
  }
}
