'use strict';

const { Matrix, inverse } = require('ml-matrix');

const selection = require('./array-utils/selection');

function setDifference(a, b) {
  let c = [];
  for (let i of a) {
    if (!b.includes(i)) c.push(i);
  }
  return c;
}
