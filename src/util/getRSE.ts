import { Matrix } from 'ml-matrix';

/**
 * Return the root square error of the solution.
 * @param object - with X, K, Y, and error @see {@link GetRSEInput}
 * @param object.X
 * @param object.K
 * @param object.Y
 * @param object.error
 * @returns the root squared error array.
 */
export function getRSE({ X, K, Y, error }: GetRSEInput) {
  error = X.mmul(K).sub(Y).pow(2);
  const sumRows = error.getRowVector(0);
  for (let i = 1; i < error.rows; i++) {
    sumRows.add(error.getRowVector(i));
  }
  return sumRows.sqrt();
}

interface GetRSEInput {
  X: Matrix;
  K: Matrix;
  Y: Matrix;
  error: Matrix;
}
