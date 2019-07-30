'use strict';

const selection = require('../selection.js');

describe('selection test', () => {
  it('simple', () => {
    expect(Array.from(selection([1, 2, 3, 4, 5], [2, 3]))).toStrictEqual([
      3,
      4,
    ]);
  });
});
