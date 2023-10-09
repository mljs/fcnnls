/**
 * Computes the set difference A\B
 * @param set A as an array
 * @param set B as an array
 * @returns Elements of A that are not in B
 */
export function setDifference(A: number[], B: number[]) {
  const C = [];
  for (const i of A) {
    if (!B.includes(i)) C.push(i);
  }
  return C;
}
