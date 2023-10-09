/**
 * Sorts an array and returns an object with the sorted array and the corresponding indices.
 * @param array
 * @returns {values, indices}
 */
export function sortArray(array: number[]) {
  const v = array.map((value, index) => {
    return { value, index };
  });

  v.sort((a, b) => {
    if (a.value === b.value) return a.index - b.index;
    return a.value - b.value;
  });
  const values: number[] = [];
  const indices: number[] = [];
  for (const item of v) {
    values.push(item.value);
    indices.push(item.index);
  }
  return { values, indices };
}
