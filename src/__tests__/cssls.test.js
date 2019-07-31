'use strict';

const cssls = require('../cssls.js');

describe('cssls test', () => {
  it('simple', () => {
    expect(Array.from(cssls())).toStrictEqual([3, 4]);
  });
});
