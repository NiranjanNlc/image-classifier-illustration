"""
Script to run all tests with coverage report
"""
import subprocess
import sys


def main():
    """Run pytest with coverage"""
    print("=" * 60)
    print("Running Image Classifier API Tests")
    print("=" * 60)
    print()
    
    # Run pytest with coverage
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            '--verbose',
            '--cov=.',
            '--cov-report=term-missing',
            '--cov-exclude=tests/*',
            'tests/'
        ],
        cwd='.',
    )
    
    print()
    print("=" * 60)
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed with return code: {result.returncode}")
    print("=" * 60)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())

