import { expect, it, describe } from 'vitest';

import { setDifference } from '../setDifference';

describe('setDifference test', () => {
  it('simple', () => {
    const result = setDifference([1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 6]);
    expect(result).toStrictEqual([2, 4, 7]);
  });
  it('same set', () => {
    const result = setDifference([1, 2, 3], [1, 2, 3]);
    expect(result).toStrictEqual([]);
  });
});
