'use strict';

const sortCollectionSet = require('../sortCollectionSet');

describe('sortCollectionSet test', () => {
  it('collection of 1 set', () => {
    let result = sortCollectionSet([[1, 2, 3]]);
    expect(result).toStrictEqual({
      values: [[1, 2, 3]],
      indices: [[0]],
    });
  });
  it('collection of 4 set', () => {
    let result = sortCollectionSet([
      [1, 2, 3],
      [2, 4, 5],
      [1, 6, 8, 10],
      [1, 2, 3],
    ]);
    expect(result).toStrictEqual({
      values: [[1, 2, 3], [2, 4, 5], [1, 6, 8, 10]],
      indices: [[0, 3], [1], [2]],
    });
  });
  it('collection of 10 set', () => {
    let result = sortCollectionSet([
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
});
