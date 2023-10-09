import { expect, it, describe } from 'vitest';

import { sortArray } from '../sortArray';

describe('sortArray test', () => {
  it('simple 3 elements array', () => {
    const result = sortArray([2, 1, 3, 7, -10, -11, -109, 0]);
    expect(result).toStrictEqual({
      values: [-109, -11, -10, 0, 1, 2, 3, 7],
      indices: [6, 5, 4, 7, 1, 0, 2, 3],
    });
  });
});
