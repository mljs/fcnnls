export default {
  input: 'src/index.js',
  output: {
    file: 'fcnnls.js',
    format: 'cjs',
    exports: 'named',
  },
  external: ['ml-matrix'],
};
