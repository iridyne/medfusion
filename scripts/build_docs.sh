#!/bin/bash
# Build Sphinx documentation

set -e

echo "=========================================="
echo "Building MedFusion Documentation"
echo "=========================================="
echo

# Check if sphinx is installed
if ! command -v sphinx-build &> /dev/null; then
    echo "❌ sphinx-build not found. Installing documentation dependencies..."
    pip install sphinx sphinx-rtd-theme myst-parser
fi

# Navigate to docs directory
cd "$(dirname "$0")/../docs"

echo "📁 Working directory: $(pwd)"
echo

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf _build
echo

# Build HTML documentation
echo "🔨 Building HTML documentation..."
sphinx-build -b html . _build/html -W --keep-going
echo

# Check if build was successful
if [ -f "_build/html/index.html" ]; then
    echo "=========================================="
    echo "✅ Documentation built successfully!"
    echo "=========================================="
    echo
    echo "📄 Documentation location: docs/_build/html/index.html"
    echo
    echo "To view the documentation:"
    echo "  1. Open in browser: file://$(pwd)/_build/html/index.html"
    echo "  2. Or run: python -m http.server 8000 -d _build/html"
    echo "     Then visit: http://localhost:8000"
    echo
else
    echo "=========================================="
    echo "❌ Documentation build failed!"
    echo "=========================================="
    exit 1
fi
