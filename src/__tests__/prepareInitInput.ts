import { Matrix } from 'ml-matrix';

export function prepareInput(X: Matrix, Y: Matrix) {
  const Xt = X.transpose();
  const XtX = Xt.mmul(X);
  const XtY = Xt.mmul(Y);
  const { columns: nColsY, rows: nRowsY } = Y;
  const { columns: nColsX, rows: nRowsX } = X;
  return { XtX, XtY, nRowsX, nColsX, nRowsY, nColsY };
}
