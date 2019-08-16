# fcnnls

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Fast Combinatorial Non-negative Least Squares.

Fast algorithm for the solution of large‐scale non‐negativity‐constrained least squares problems from Van Benthem and Keenan ([10.1002/cem.889](http://doi.org/10.1002/cem.889)), based on the active-set method algorithm published by Lawson and Hanson.

It solves the following optimisation problem.
Given <img src='images/Im1.svg'> an <img src='images/Im2.svg'> matrix and <img src='images/Im3.svg'> an <img src='images/Im4.svg'> matrix, find <img src='images/Im5.svg'> subject to <img src='images/Im6.svg'>, where <img src='images/Im7.svg'> is an <img src='images/Im8.svg'> matrix and <img src='images/Im9.svg'> is the Frobenius norm. In fact, <img src='images/Im7.svg'> is the best solution to the equation: <img src='images/Im11.svg'>, where <img src='images/Im6.svg'>. Note that if <img src='images/Im3.svg'> is a column vector, it performs the regular Non-negative Least Squares algorithm and finds a vector as a solution to the problem. Also, performing this algorithm when <img src='images/Im3.svg'> is a matrix is like running the algorithm on each column of <img src='images/Im3.svg'>, it will give the same result but in a much more efficient way.

https://en.wikipedia.org/wiki/Non-negative_least_squares

## Installation

`$ npm i ml-fcnnls`

## [API Documentation](https://mljs.github.io/fcnnls/)

## Usage

```js
import { Matrix } from 'ml-matrix';
import { fcnnls } from 'index';

// Example with multiple RHS

let X = new Matrix([[1, 1, 2], [10, 11, -9], [-1, 0, 0], [-5, 6, -7]]);

// Y can either be a Matrix of an array of array
let Y = new Matrix([
  [-1, 0, 0, 9],
  [11, -20, 103, 5],
  [0, 0, 0, 0],
  [1, 2, 3, 4],
]);

let K = fcnnls(X, Y);

/*
K = Matrix([
  [0.461, 0, 4.9714, 0],
  [0.5611, 0, 4.7362, 2.2404],
  [0, 1.2388, 0, 1.9136],
    ])
*/

import { fcnnlsVector } from 'ml-fcnnls';

// Example with single RHS and same X
// Should be giving a vector with the element of the first column of K in the previous example, since y is the first column of Y

let X = new Matrix([[1, 1, 2], [10, 11, -9], [-1, 0, 0], [-5, 6, -7]]);

let y = [-1, 11, 0, 1];

let k = fcnnlsVector(X, y);

/*
k = [0.461, 0.5611, 0]
*/
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-fcnnls.svg?style=flat-square
[npm-url]: https://www.npmjs.com/package/ml-fcnnls
[travis-image]: https://img.shields.io/travis/com/mljs/fcnnls/master.svg?style=flat-square
[travis-url]: https://travis-ci.com/mljs/fcnnls
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/fcnnls.svg?style=flat-square
[codecov-url]: https://codecov.io/gh/mljs/fcnnls
[download-image]: https://img.shields.io/npm/dm/ml-fcnnls.svg?style=flat-square
[download-url]: https://www.npmjs.com/package/ml-fcnnls
