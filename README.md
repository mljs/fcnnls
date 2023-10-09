# fcnnls

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8189402.svg)](https://doi.org/10.5281/zenodo.8189402)

Fast Combinatorial Non-negative Least Squares.

As described in the publication by Van Benthem and Keenan ([10.1002/cem.889](http://doi.org/10.1002/cem.889)), which is in turn based on the active-set method algorithm previously published by Lawson and Hanson. The basic active-set method is implemented in the [nnls repository.](https://github.com/mljs/nnls)

Given the matrices $\mathbf{X}$ and $\mathbf{Y}$, the code finds the matrix $\mathbf{K}$ that minimises the squared Frobenius norm $$\mathrm{argmin}_K ||\mathbf{XK} -\mathbf{Y}||^2_F$$ subject to $\mathbf{K}\geq 0$.

https://en.wikipedia.org/wiki/Non-negative_least_squares

## Installation

```bash
npm i ml-fcnnls
```

## Usage Example

1. Single $y$, using arrays as inputs.

```js
import { fcnnlsVector } from 'ml-fcnnls';

const X = [
  [1, 1, 2],
  [10, 11, -9],
  [-1, 0, 0],
  [-5, 6, -7],
];
const y = [-1, 11, 0, 1];

const k = fcnnlsVector(X, y).K.to1DArray();
/* k = [0.4610, 0.5611, 0] */
```

2. Multiple RHS, using `Matrix` instances as inputs.

```js
import { fcnnls } from 'ml-fcnnls';
import { Matrix } from 'ml-matrix'; //npm i ml-matrix

// Example with multiple RHS

const X = new Matrix([
  [1, 1, 2],
  [10, 11, -9],
  [-1, 0, 0],
  [-5, 6, -7],
]);

// Y can either be a Matrix or an array of arrays
const Y = new Matrix([
  [-1, 0, 0, 9],
  [11, -20, 103, 5],
  [0, 0, 0, 0],
  [1, 2, 3, 4],
]);

const K = fcnnls(X, Y).K;
// `K.to2DArray()` converts the matrix to array.
/*
K = Matrix([
  [0.4610, 0, 4.9714, 0],
  [0.5611, 0, 4.7362, 2.2404],
  [0, 1.2388, 0, 1.9136],
])
*/
```

3. Using the options

```js
const K = fcnnls(X, Y, {
  info: true, // returns the error/iteration.
  maxIterations: 5,
  gradientTolerance: 0,
});
/* same result than 2*/
```

## [API Documentation](https://mljs.github.io/fcnnls/)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-fcnnls.svg
[npm-url]: https://www.npmjs.com/package/ml-fcnnls
[ci-image]: https://github.com/mljs/fcnnls/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/mljs/fcnnls/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/fcnnls.svg
[codecov-url]: https://codecov.io/gh/mljs/fcnnls
[download-image]: https://img.shields.io/npm/dm/ml-fcnnls.svg
[download-url]: https://www.npmjs.com/package/ml-fcnnls
