'use strict';

const sortArray = require('../sortArray.js');

describe('sortArray test', () => {
  it('simple 3 elements array', () => {
    let result = sortArray([2, 1, 3]);
    expect(result).toStrictEqual({
      values: [1, 2, 3],
      indices: [1, 0, 2],
    });
  });
});
