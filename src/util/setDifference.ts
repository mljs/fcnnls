/**
 * Computes the set difference A\B
 * @param A - First array of numbers
 * @param B - Second array of numbers
 * @returns Elements of A that are not in B
 */
export function setDifference(A: number[], B: number[]) {
  const C = [];
  for (const i of A) {
    if (!B.includes(i)) C.push(i);
  }
  return C;
}
