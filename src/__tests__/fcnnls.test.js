'use strict';

const {
  toBeDeepCloseTo,
  toMatchCloseTo,
} = require('jest-matcher-deep-close-to');

expect.extend({ toBeDeepCloseTo, toMatchCloseTo });

const { Matrix } = require('ml-matrix');

const fcnnls = require('../fcnnls');

const { readFileSync } = require('fs');
const { join } = require('path');

const concentration = readFileSync(join(__dirname, 'data/matrix.txt'), 'utf-8');
let linesA = concentration.split(/[\r\n]+/);
let A = [];
for (let line of linesA) {
  A.push(line.split(',').map((value) => Number(value)));
}

let matrix = new Matrix(A);

matrix = matrix.transpose();

const proportion = readFileSync(join(__dirname, 'data/x_fcnnls.txt'), 'utf-8');
let linesk = proportion.split(/[\r\n]+/);
let k = [];
for (let line of linesk) {
  k.push(line.split(',').map((value) => Number(value)));
}
delete k.splice(133, 1);
let answer = new Matrix(k);

const observation = readFileSync(join(__dirname, 'data/target.txt'), 'utf-8');
let lines = observation.split(/[\r\n]+/);
let b = [];
for (let line of lines) {
  b.push(line.split(',').map((value) => Number(value)));
}

let target = new Matrix(b);

target = target.transpose();

describe('myModule test', () => {
  it('identity X, Y 4x1', () => {
    let X = Matrix.eye(4);
    let Y = new Matrix([[0], [1], [2], [3]]);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it('identity X, Y 5x3', () => {
    let X = Matrix.eye(5);
    let Y = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    let solution = new Matrix([
      [0, 5, 10],
      [1, 6, 11],
      [2, 7, 12],
      [3, 8, 13],
      [4, 9, 14],
    ]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it('non-singular square X, Y 3x1', () => {
    let X = new Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]]);
    let Y = new Matrix([[-1], [2], [-3]]);
    let solution = new Matrix([[0], [0], [0.5]]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it('singular square X rank 2, Y 3x1', () => {
    let X = new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let Y = new Matrix([[-1], [0], [10]]);
    let solution = new Matrix([[1.0455], [0], [0]]);
    let result = Matrix.round(fcnnls(X, Y).mul(10000)).mul(0.0001);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it('6x3 X full-rank, Y 6x7', () => {
    let X = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [0, 1, 1],
    ]);
    let Y = new Matrix([
      [-1, 0, 1, 2, 3, 4, 5],
      [0, 3, 5, 6, 79, 3, 1],
      [10, 11, 2, 3, 4, 7, 8],
      [1, 112, 0, 0, 0, 7, 8],
      [1000, 2, 56, 40, 1, 1, 3],
      [7, 6, 5, 4, 3, 2, 1],
    ]);
    let solution = new Matrix([
      [39.0418, 1.3439, 2.2776, 1.6925, 0, 0, 0],
      [0, 2.121, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1.0827, 0.3911, 0.4738],
    ]);
    let result = Matrix.round(fcnnls(X, Y, false).mul(10000)).mul(0.0001);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });

  it('Van Benthem - Keenan example', () => {
    let X = new Matrix([[95, 89, 82], [23, 76, 44], [61, 46, 62], [42, 2, 79]]);
    let Y = new Matrix([
      [92, 99, 80],
      [74, 19, 43],
      [18, 41, 51],
      [41, 61, 39],
    ]);
    let solution = new Matrix([
      [0, 0.6873, 0.2836],
      [0.6272, 0, 0.2862],
      [0.3517, 0.2873, 0.335],
    ]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it('matrix/target', () => {
    let X = matrix;
    let Y = target;
    let result = fcnnls(X, Y);
    let solution = answer;
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 0);
  });

  it('example documentation', () => {
    let X = new Matrix([[1, 1, 2], [10, 11, -9], [-1, 0, 0], [-5, 6, -7]]);
    let Y = new Matrix([
      [-1, 0, 0, 9],
      [11, -20, 103, 5],
      [0, 0, 0, 0],
      [1, 2, 3, 4],
    ]);
    let solution = new Matrix([
      [0.461, 0, 4.9714, 0],
      [0.5611, 0, 4.7362, 2.2404],
      [0, 1.2388, 0, 1.9136],
    ]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toMatchCloseTo(solution.to2DArray(), 4);
  });

  it.skip('random  X, ramdom Y', () => {
    let X = Matrix.randInt(10000, 20);
    let Y = Matrix.randInt(10000, 200);
    //console.log(X);
    //console.log(Y);
    let solution = new Matrix([[0], [1], [2], [3]]);
    let result = fcnnls(X, Y, false);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });

  it('debuggage random matrices', () => {
    let X = new Matrix([
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

    let Y = new Matrix([
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
    let solution = new Matrix([
      [0.2066, 0.7494],
      [0.2479, 0.5763],
      [0.0004, 0],
      [0, 0],
      [0.492, 0],
    ]);
    let result = fcnnls(X, Y);
    expect(result.to2DArray()).toBeDeepCloseTo(solution.to2DArray(), 4);
  });
});
