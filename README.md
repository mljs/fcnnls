# fcnnls

Fast Combinatorial Non-negative Least Squares

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![David deps][david-image]][david-url]
[![npm download][download-image]][download-url]

Fast algorithm for the solution of large‐scale non‐negativity‐constrained least squares problems from Van Benthem and Keenan ([10.1002/cem.889](http://doi.org/10.1002/cem.889)), based on the active-set method algorithm published by Lawson and Hanson.

It solves the following optimisation problem.
Given <img src='images/Im1.svg'> an <img src='images/Im2.svg'> matrix and <img src='images/Im3.svg'> an <img src='images/Im4.svg'> matrix, find <img src='images/Im5.svg'> subject to <img src='images/Im6.svg'>, where <img src='images/Im7.svg'> is an <img src='images/Im8.svg'> matrix and <img src='images/Im9.svg'> is the Frobenius norm. 

https://en.wikipedia.org/wiki/Non-negative_least_squares

## Installation

`$ npm i fcnnls`

## [API Documentation](https://cheminfo.github.io/fcnnls/)

## Example

This first example 

```js
const { Matrix } = require('ml-matrix');
const { fcnnls }  = require('fcnnls');

let X = new Matrix([
  [1, 1, 2], 
  [10, 11, -9], 
  [-1, 0, 0], 
  [-5, 6, -7],
]);

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

```



```
const {fcnnls}  = require('fcnnls');

let y = [-1, 11, 0, 1];


let k = fcnnlsVector(X, y);

let K = fcnnls(X, Y);


/*
k = [0.461, 0.5611, 0]
*/





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
