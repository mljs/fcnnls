'use strict';

// added "eslint-plugin-babel" in json but does not work...

let bigInt = require('big-integer');

/**
 *
 * @param {Array of arrays} collection
 */

function sortCollectionSet(collection) {
  let objectCollection = collection
    .map((value, index) => {
      let key = bigInt(0); // 0n
      value.forEach((item) => (key |= bigInt(1) << bigInt(item))); // 1n BigInt(item)
      return { value, index, key };
    })
    .sort((a, b) => {
      if (a.key - b.key < 0) return -1;
      return 1;
    });

  let sorted = [];
  let indices = [];

  let key;
  for (let set of objectCollection) {
    if (JSON.stringify(set.key) !== JSON.stringify(key)) {
      //set.key !== key
      key = set.key;
      indices.push([]);
      sorted.push(set.value);
    }
    indices[indices.length - 1].push(set.index);
  }

  let result = {
    values: sorted,
    indices: indices,
  };

  return result;
}

module.exports = sortCollectionSet;
