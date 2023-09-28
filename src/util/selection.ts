/**
 * Returns a new array based on extraction of specific indices of an array
 * @param collection or array
 * @param indices
 */
export function selection<T extends number[] | number[][]>(
  vector: T,
  indices: number[],
) {
  const u: T = [] as unknown as T; //new Float64Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    u[i] = vector[indices[i]];
  }
  return u;
}
