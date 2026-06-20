import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { fcnnls } from '../fcnnls.ts';

import { X as X1, Y as Y1 } from './deconv-examples/data_example1.ts';
import { X as X2, Y as Y2 } from './deconv-examples/data_example2.ts';
import { X as X3, Y as Y3 } from './deconv-examples/data_example3.ts';

test('example 1', () => {
  const result = fcnnls(X1, Y1);

  expect(result).toBeDefined();
});

test('example 2', () => {
  const result = fcnnls(X2, Y2, { gradientTolerance: 1e-4 });

  expect(result).toBeDefined();
});

test('example 3', () => {
  const result = fcnnls(X3, Y3, { gradientTolerance: 1e-4 });

  expect(result).toBeDefined();
});

test('example 4: X3 - 1', () => {
  const X4 = new Matrix(X3).sub(1);
  const Y4 = new Matrix(Y3).sub(1);
  const result = fcnnls(X4, Y4);

  expect(result).toBeDefined();
});
