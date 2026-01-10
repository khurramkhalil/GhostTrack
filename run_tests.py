"""
Test runner for GhostTrack.

Runs all tests and reports coverage.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_all_tests(verbose=True):
    """
    Run all tests in the tests directory.

    Args:
        verbose: Whether to print detailed test output.

    Returns:
        True if all tests passed, False otherwise.
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    # Return success status
    return result.wasSuccessful()


def run_specific_test(test_module: str, verbose=True):
    """
    Run a specific test module.

    Args:
        test_module: Name of test module (e.g., 'test_config').
        verbose: Whether to print detailed output.

    Returns:
        True if tests passed, False otherwise.
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run GhostTrack tests')
    parser.add_argument(
        '--module',
        type=str,
        help='Specific test module to run (e.g., test_config)',
        default=None
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Run tests with minimal output'
    )

    args = parser.parse_args()

    if args.module:
        success = run_specific_test(args.module, verbose=not args.quiet)
    else:
        success = run_all_tests(verbose=not args.quiet)

    sys.exit(0 if success else 1)
