'use strict';

/**
 *
 * @param {Array<Number>} v
 */
function sortArray(array) {
  const v = array.map((value, index) => {
    return { value, index };
  });

  v.sort((a, b) => {
    if (a.value === b.value) return a.index - b.index;
    return a.value - b.value;
  });

  let values = v.map((item) => item.value);
  let indices = v.map((item) => item.index);
  return { values, indices };
}

module.exports = sortArray;
