import { expect, it, describe } from 'vitest';

import { diff } from '../diff';

describe('diff test', () => {
  it('simple', () => {
    const result = diff([1, 2, 3, 7, 36, -159, 0]);
    expect(result).toStrictEqual([1, 1, 4, 29, -195, 159]);
  });
});
