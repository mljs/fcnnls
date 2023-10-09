import { expect, it, describe } from 'vitest';

import { sortCollectionSet } from '../sortCollectionSet';

describe('sortCollectionSet test', () => {
  it('collection of 1 set', () => {
    const result = sortCollectionSet([[1, 2, 3]]);
    expect(result).toStrictEqual({
      values: [[1, 2, 3]],
      indices: [[0]],
    });
  });
  it('collection of 4 set', () => {
    const result = sortCollectionSet([
      [1, 2, 3],
      [2, 4, 5],
      [1, 6, 8, 10],
      [1, 2, 3],
    ]);
    expect(result).toStrictEqual({
      values: [
        [1, 2, 3],
        [2, 4, 5],
        [1, 6, 8, 10],
      ],
      indices: [[0, 3], [1], [2]],
    });
  });
  it('collection of 10 set', () => {
    const result = sortCollectionSet([
      [1, 2],
      [2, 4, 5],
      [1, 6, 8, 10],
      [1, 2, 3],
      [],
      [],
      [],
      [120],
      [1, 2, 3, 4, 5, 56, 8, 90],
    ]);
    expect(result).toStrictEqual({
      values: [
        [],
        [1, 2],
        [1, 2, 3],
        [2, 4, 5],
        [1, 6, 8, 10],
        [1, 2, 3, 4, 5, 56, 8, 90],
        [120],
      ],
      indices: [[4, 5, 6], [0], [3], [1], [2], [8], [7]],
    });
  });
  it('collection of 1 empty set', () => {
    const result = sortCollectionSet([[]]);
    expect(result).toStrictEqual({
      values: [[]],
      indices: [[0]],
    });
  });
});
