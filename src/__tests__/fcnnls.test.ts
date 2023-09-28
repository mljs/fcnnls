import { readFileSync } from 'fs';
import { join } from 'path';

import { Matrix } from 'ml-matrix';
import { it, describe } from 'vitest';

import { fcnnls } from '../fcnnls';

import { assertResult } from './assertResult';

const concentration = readFileSync(join(__dirname, 'data/matrix.txt'), 'utf-8');
const linesA = concentration.split(/[\r\n]+/);
const A: number[][] = [];
for (const line of linesA) {
  A.push(line.split(',').map((value) => Number(value)));
}

let matrix = new Matrix(A);

matrix = matrix.transpose();

const proportion = readFileSync(join(__dirname, 'data/x_fcnnls.txt'), 'utf-8');
const linesk = proportion.split(/[\r\n]+/);
const k: number[][] = [];
for (const line of linesk) {
  k.push(line.split(',').map((value) => Number(value)));
}
k.splice(133, 1);
const answer = new Matrix(k);

const observation = readFileSync(join(__dirname, 'data/target.txt'), 'utf-8');
const lines = observation.split(/[\r\n]+/);
const b: number[][] = [];
for (const line of lines) {
  b.push(line.split(',').map((value) => Number(value)));
}

let target = new Matrix(b);

target = target.transpose();
describe('Test Fast Combinatorial NNLS', () => {
  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('identity X, Y 5x3', () => {
    const X = Matrix.eye(5);
    const Y = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    const solution = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('non-singular square X, Y 3x1', () => {
    const X = new Matrix([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
    const Y = new Matrix([[-1], [2], [-3]]);
    const solution = new Matrix([[0], [0], [0.5]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('singular square X rank 2, Y 3x1', () => {
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const Y = new Matrix([[-1], [0], [10]]);
    const solution = new Matrix([[1.0455], [0], [0]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('6x3 X full-rank, Y 6x7', () => {
    const X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [0, 1, 1],
    ]);
    const Y = new Matrix([
      [-1, 0, 1, 2, 3, 4, 5],
      [0, 3, 5, 6, 79, 3, 1],
      [10, 11, 2, 3, 4, 7, 8],
      [1, 112, 0, 0, 0, 7, 8],
      [1000, 2, 56, 40, 1, 1, 3],
      [7, 6, 5, 4, 3, 2, 1],
    ]);
    const solution = new Matrix([
      [39.0418, 1.3439, 2.2776, 1.6925, 0, 0, 0],
      [0, 2.121, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1.0827, 0.3911, 0.4738],
    ]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('Van Benthem - Keenan example', () => {
    const X = new Matrix([
      [95, 89, 82],
      [23, 76, 44],
      [61, 46, 62],
      [42, 2, 79],
    ]);
    const Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);
    const solution = new Matrix([
      [0, 0.6873, 0.2836],
      [0.6272, 0, 0.2862],
      [0.3517, 0.2873, 0.335],
    ]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('matrix/target', () => {
    const X = matrix;
    const Y = target;
    const result = fcnnls(X, Y);
    const solution = answer;
    assertResult(result, solution);
  });

  it('example documentation', () => {
    const X = new Matrix([
      [1, 1, 2],
      [10, 11, -9],
      [-1, 0, 0],
      [-5, 6, -7],
    ]);
    const Y = new Matrix([
      [-1, 0, 0, 9],
      [11, -20, 103, 5],
      [0, 0, 0, 0],
      [1, 2, 3, 4],
    ]);
    const solution = new Matrix([
      [0.461, 0, 4.9714, 0],
      [0.5611, 0, 4.7362, 2.2404],
      [0, 1.2388, 0, 1.9136],
    ]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('debuggage random matrices', () => {
    const X = new Matrix([
      [269, 134, 116, 940, 530],
      [899, 199, 207, 95, 533],
      [474, 817, 100000, 743, 991],
      [315, 389, -90, 950, 169],
      [963, 742, 151, 501, 282],
      [777, 618, 991, 738, 969],
      [177, 460, 268, 210, 106],
      [866, 802, 504, 665, 896],
      [397, 832, 951, 242, 359],
      [203, 763, 405, 179, 454],
    ]);

    const Y = new Matrix([
      [123, 521],
      [950, 21],
      [825, 657],
      [192, 151],
      [280, 3089],
      [424, 88],
      [226, 748],
      [997, 845],
      [309, 163],
      [902, 629],
    ]);
    const solution = new Matrix([
      [0.2066, 0.7494],
      [0.2479, 0.5763],
      [0.0004, 0],
      [0, 0],
      [0.492, 0],
    ]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('identity X, negative Y 3x1', () => {
    const X = Matrix.eye(3);
    const Y = new Matrix([[-1], [-2], [-3]]);
    const solution = new Matrix([[0], [0], [0]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('another simple test', () => {
    const X = new Matrix([
      [1, 1, 0],
      [0, 1, 1],
      [0, 0, 1],
    ]);
    const Y = new Matrix([[-2], [2], [0]]);
    const solution = new Matrix([[0], [0], [1]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });

  it('non positive-definite matrix', () => {
    const X = new Matrix([
      [1, 1, 1, 0],
      [0, 1, 1, 1],
      [1, 2, 2, 1],
    ]);
    const Y = new Matrix([[-2], [2], [0]]);
    const solution = new Matrix([[0], [0], [0], [1]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });
  it('identity X, Y 4x1', () => {
    const X = Matrix.eye(4);
    const Y = new Matrix([[0], [1], [2], [3]]);
    const solution = new Matrix([[0], [1], [2], [3]]);
    const result = fcnnls(X, Y);
    assertResult(result, solution);
  });
});
