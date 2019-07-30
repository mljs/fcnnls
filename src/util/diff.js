'use strict';

const { Matrix, inverse } = require('ml-matrix');

const selection = require('./array-utils/selection');

/**
 *
 * @param {Array} v
 */
function diff(v) {
  let u = [];
  for (let i = 0; i < v.length - 1; i++) {
    u.push(v[i + 1] - v[i]);
  }
  return u;
}
