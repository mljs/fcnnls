import { expect, test } from 'vitest';

import { selection } from '../selection.ts';

test('simple', () => {
  expect(Array.from(selection([1, 2, 3, 4, 5], [2, 3]))).toStrictEqual([3, 4]);
});

test('very simple', () => {
  expect(Array.from(selection([[1, 2, 3]], [0]))).toStrictEqual([[1, 2, 3]]);
});
