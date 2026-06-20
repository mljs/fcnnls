import { defineConfig, globalIgnores } from 'eslint/config';
import ts from 'eslint-config-cheminfo-typescript';

export default defineConfig(
  globalIgnores(['coverage', 'lib']),
  ts,
  {
    // The benchmark is a profiling script; console output is its purpose.
    files: ['benchmark/**'],
    rules: {
      'no-console': 'off',
    },
  },
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
