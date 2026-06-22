import type { Matrix } from 'ml-matrix';

/**
 * Return the root square error of the solution: for each column of Y, the
 * Euclidean norm of the residual X·K − Y.
 * @param object - with X, K and Y. {@link GetRSEInput}
 * @param object.X - The data/input/predictors matrix.
 * @param object.K - The coefficients matrix.
 * @param object.Y - The response matrix.
 * @returns the root squared error as a row vector (one value per column of Y).
 */
export function getRSE({ X, K, Y }: GetRSEInput) {
  const squaredError = X.mmul(K).sub(Y).pow(2);
  const sumRows = squaredError.getRowVector(0);
  for (let i = 1; i < squaredError.rows; i++) {
    sumRows.add(squaredError.getRowVector(i));
  }
  return sumRows.sqrt();
}

interface GetRSEInput {
  X: Matrix;
  K: Matrix;
  Y: Matrix;
}
