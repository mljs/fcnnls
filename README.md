# fcnnls

Fast Combinatorial Non-negative Least Squares

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![David deps][david-image]][david-url]
[![npm download][download-image]][download-url]

Fast algorithm for the solution of large‐scale non‐negativity‐constrained least squares problems from Van Benthem and Keenan ([10.1002/cem.889](http://doi.org/10.1002/cem.889)), based on the active-set method algorithm published by Lawson and Hanson.
It solves the following optimisation problem.
Given $ X $ an $ n \times l $ matrix and $ Y $ an $ n \times p $ matrix, find $ argmin_K\|XK - Y\|_{F}^{2} $ subject to $ K \geq 0 $, where $ K $ is an $ l \times p $ matrix and $ \| . \|_{F} $ is the Frobenius norm.

https://en.wikipedia.org/wiki/Non-negative_least_squares

## Installation

`$ npm i fcnnls`

## [API Documentation](https://cheminfo.github.io/fcnnls/)

## Example

```js
const fcnnls = require('fcnnls');
const { Matrix } = require('ml-matrix');

let X = new Matrix([[1, 1, 2], [10, 11, -9], [-1, 0, 0], [-5, 6, -7]]);
let Y = new Matrix([
  [-1, 0, 0, 9],
  [11, -20, 103, 5],
  [0, 0, 0, 0],
  [1, 2, 3, 4],
]);
let K = fcnnls(X, Y);

/*
K=
*/
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/fcnnls.svg?style=flat-square
[npm-url]: https://www.npmjs.com/package/fcnnls
[travis-image]: https://img.shields.io/travis/com/cheminfo/fcnnls/master.svg?style=flat-square
[travis-url]: https://travis-ci.com/cheminfo/fcnnls
[codecov-image]: https://img.shields.io/codecov/c/github/cheminfo/fcnnls.svg?style=flat-square
[codecov-url]: https://codecov.io/gh/cheminfo/fcnnls
[david-image]: https://img.shields.io/david/cheminfo/fcnnls.svg?style=flat-square
[david-url]: https://david-dm.org/cheminfo/fcnnls
[download-image]: https://img.shields.io/npm/dm/fcnnls.svg?style=flat-square
[download-url]: https://www.npmjs.com/package/fcnnls
