/**
 * From an array of arrays it constructs a unique key for each array,
 * given by its values, such that same arrays have same keys.
 *
 * @param collection - Array of arrays
 * @returns - Array of objects with the original array, its index and its key
 */
function addUniqueKeyToColumns(collection: number[][]) {
  return collection.map((positiveRows, columnIndexInK) => {
    //indices of positive values within the column. (Pset)
    let key = BigInt(0);
    // items will be the indexes of Pset, so it's always an integer.
    positiveRows.forEach((item) => (key |= BigInt(1) << BigInt(item)));
    return { positiveRows, columnIndexInK, key };
  });
}

/**
 * From an array of arrays it constructs a unique key for each subarray,
 * given by its values, such that same arrays have same keys.
 * @param collection - Array of arrays
 * @returns Array of objects with the original array, its index and its key
 */
export function sortCollectionSet(collection: number[][]) {
  const mapped = addUniqueKeyToColumns(collection);
  mapped.sort((a, b) => {
    if (a.key - b.key < 0) return -1;
    return 1;
  });

  const sorted: number[][] = [];
  const indices: number[][] = [];

  let key;
  for (const set of mapped) {
    if (set.key !== key) {
      key = set.key;
      indices.push([]);
      sorted.push(set.positiveRows);
    }
    indices[indices.length - 1].push(set.columnIndexInK);
  }

  const result = {
    values: sorted,
    indices,
  };
  return result;
}
