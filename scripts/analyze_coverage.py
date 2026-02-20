#!/usr/bin/env python3
"""
Test coverage analysis script for MedFusion.
Identifies modules that need more test coverage.
"""

import os
import sys
from pathlib import Path


def find_python_files(directory):
    """Find all Python files in a directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]

        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def find_test_files(directory):
    """Find all test files."""
    test_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    return sorted(test_files)


def get_module_name(filepath, base_dir):
    """Convert file path to module name."""
    rel_path = os.path.relpath(filepath, base_dir)
    module = rel_path.replace(os.sep, '.').replace('.py', '')
    return module


def analyze_coverage():
    """Analyze test coverage."""
    base_dir = Path(__file__).parent.parent
    med_core_dir = base_dir / 'med_core'
    tests_dir = base_dir / 'tests'

    print("=" * 80)
    print("MedFusion Test Coverage Analysis")
    print("=" * 80)
    print()

    # Find all source files
    source_files = find_python_files(med_core_dir)
    test_files = find_test_files(tests_dir)

    print(f"📁 Source files: {len(source_files)}")
    print(f"🧪 Test files: {len(test_files)}")
    print()

    # Categorize by module
    modules = {}
    for filepath in source_files:
        parts = Path(filepath).relative_to(med_core_dir).parts
        if len(parts) > 0:
            module = parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(filepath)

    # Categorize test files
    test_modules = {}
    for filepath in test_files:
        filename = os.path.basename(filepath)
        # Extract module name from test file
        # e.g., test_backbones.py -> backbones
        if filename.startswith('test_'):
            module = filename[5:].replace('.py', '')
            if module not in test_modules:
                test_modules[module] = []
            test_modules[module].append(filepath)

    print("📊 Coverage by Module:")
    print("-" * 80)
    print(f"{'Module':<30} {'Files':<10} {'Tests':<10} {'Status':<20}")
    print("-" * 80)

    total_files = 0
    total_with_tests = 0

    for module, files in sorted(modules.items()):
        file_count = len(files)
        test_count = len(test_modules.get(module, []))

        total_files += file_count
        if test_count > 0:
            total_with_tests += 1

        if test_count == 0:
            status = "❌ No tests"
        elif test_count < file_count:
            status = "⚠️  Partial coverage"
        else:
            status = "✅ Good coverage"

        print(f"{module:<30} {file_count:<10} {test_count:<10} {status:<20}")

    print("-" * 80)
    coverage_pct = (total_with_tests / len(modules) * 100) if modules else 0
    print(f"Overall module coverage: {total_with_tests}/{len(modules)} ({coverage_pct:.1f}%)")
    print()

    # Identify modules needing tests
    print("🎯 Modules Needing More Tests:")
    print("-" * 80)

    needs_tests = []
    for module, files in sorted(modules.items()):
        test_count = len(test_modules.get(module, []))
        if test_count == 0:
            needs_tests.append((module, len(files)))

    if needs_tests:
        for module, file_count in needs_tests:
            print(f"  • {module} ({file_count} files)")
    else:
        print("  All modules have some test coverage! 🎉")

    print()

    # List existing test files
    print("📝 Existing Test Files:")
    print("-" * 80)
    for test_file in test_files:
        rel_path = os.path.relpath(test_file, base_dir)
        print(f"  • {rel_path}")

    print()
    print("=" * 80)

    return needs_tests


if __name__ == '__main__':
    needs_tests = analyze_coverage()

    if needs_tests:
        print(f"\n⚠️  Found {len(needs_tests)} modules without tests")
        sys.exit(1)
    else:
        print("\n✅ All modules have test coverage!")
        sys.exit(0)
