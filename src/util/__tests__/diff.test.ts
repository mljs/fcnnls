import { expect, test } from 'vitest';

import { diff } from '../diff.ts';

test('simple', () => {
  const result = diff([1, 2, 3, 7, 36, -159, 0]);

  expect(result).toStrictEqual([1, 1, 4, 29, -195, 159]);
});
