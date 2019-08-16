'use strict';

/**
 * Computes an array containing the difference of consecutives numbers
 * @param {Array} Array v from which it computes the difference
 */
function diff(v) {
  let u = [];
  for (let i = 0; i < v.length - 1; i++) {
    u.push(v[i + 1] - v[i]);
  }
  return u;
}

module.exports = diff;
