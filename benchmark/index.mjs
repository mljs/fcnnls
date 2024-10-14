/*
Can be executed using `tsx --inspect-brk benchmark/index.js`
Or `tsx --cpu-prof benchmark/index.js`
And debug from chrome using `chrome://inspect`
*/

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { Matrix } from 'ml-matrix';

import { fcnnls } from '../src/fcnnls';

const __dirname = join(new URL(import.meta.url).pathname, '..');

const concentration = readFileSync(
  join(__dirname, '../src/__tests__/data/matrix.txt'),
  'utf8',
);
let linesA = concentration.split(/[\r\n]+/);
let A = [];
for (let line of linesA) {
  A.push(line.split(',').map(Number));
}
let matrix = new Matrix(A);
matrix = matrix.transpose();

const observation = readFileSync(
  join(__dirname, '../src/__tests__/data/target.txt'),
  'utf8',
);
let lines = observation.split(/[\r\n]+/);
let b = [];
for (let line of lines) {
  b.push(line.split(',').map(Number));
}
let target = new Matrix(b);
target = target.transpose();

console.profile('start');
console.time('flag');
for (let i = 0; i < 20; i++) {
  let result = fcnnls(matrix, target);
}
console.timeEnd('flag');
console.profileEnd();
