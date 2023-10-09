/**
 * Computes an array containing the difference of consecutive numbers
 * @param Array v from which it computes the difference
 * @returns - Array of consecutive differences
 */
export function diff(v: number[]) {
  const u: number[] = [];
  for (let i = 0; i < v.length - 1; i++) {
    u.push(v[i + 1] - v[i]);
  }
  return u;
}
