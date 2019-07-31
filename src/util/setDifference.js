'use strict';

/**
 * Computes the set difference A\B
 * @param {A} set A as an array
 * @param {B} set B as an array
 */
function setDifference(A, B) {
  let C = [];
  for (let i of A) {
    if (!B.includes(i)) C.push(i);
  }
  return C;
}

module.exports = setDifference;
