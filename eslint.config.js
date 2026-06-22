import { defineConfig, globalIgnores } from 'eslint/config';
import ts from 'eslint-config-cheminfo-typescript';

export default defineConfig(
  // `benchmark` is gitignored local scratch (profiling scripts + vendored code).
  globalIgnores(['coverage', 'lib', 'benchmark']),
  ts,
  {
    // `assertResult` wraps the `expect` calls shared across these tests.
    files: ['**/*.test.ts'],
    rules: {
      'vitest/expect-expect': [
        'error',
        { assertFunctionNames: ['expect', 'assertResult'] },
      ],
    },
  },
);
