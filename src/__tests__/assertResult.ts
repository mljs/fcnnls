import { Matrix } from 'ml-matrix';
import { expect } from 'vitest';

import { type FcnnlsOutput } from '../fcnnls';

/**
 *  We only use the value K from the output.
 * @param output - The output of the fcnnls function
 * @param solution - The expected solution
 * @param precision - Number of digits to match
 */
export function assertResult(
  output: FcnnlsOutput,
  solution: Matrix,
  precision = 4,
) {
  const result = output.K;
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
