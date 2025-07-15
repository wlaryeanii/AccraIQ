#!/bin/bash

# AccraIQ Dependencies Installation Script
# This script installs all required dependencies for the AccraIQ dashboard

set -e  # Exit on any error

echo "🚀 Installing AccraIQ Dependencies..."

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

echo "📋 Detected OS: $OS"

# Function to install system dependencies
install_system_deps() {
    if [[ "$OS" == "linux" ]]; then
        echo "📦 Installing system dependencies for Linux..."
        
        # Detect Linux distribution
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y \
                python3-dev \
                python3-pip \
                python3-cffi \
                python3-brotli \
                libpango-1.0-0 \
                libharfbuzz0b \
                libpangoft2-1.0-0 \
                libcairo2 \
                libpangocairo-1.0-0 \
                libgdk-pixbuf2.0-0 \
                libffi-dev \
                shared-mime-info \
                chromium-browser \
                chromium-chromedriver \
                wget \
                gnupg
                
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL/Fedora
            sudo yum update -y
            sudo yum install -y \
                python3-devel \
                python3-pip \
                cairo-devel \
                pango-devel \
                gdk-pixbuf2-devel \
                libffi-devel \
                chromium \
                chromium-headless \
                chromedriver
                
        elif command -v dnf &> /dev/null; then
            # Fedora
            sudo dnf update -y
            sudo dnf install -y \
                python3-devel \
                python3-pip \
                cairo-devel \
                pango-devel \
                gdk-pixbuf2-devel \
                libffi-devel \
                chromium \
                chromium-headless \
                chromedriver
        else
            echo "❌ Unsupported Linux distribution. Please install dependencies manually."
            exit 1
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        echo "📦 Installing system dependencies for macOS..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "🍺 Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies
        brew install \
            cairo \
            pango \
            gdk-pixbuf \
            libffi \
            chromedriver
        
    elif [[ "$OS" == "windows" ]]; then
        echo "📦 Windows detected. Please install dependencies manually:"
        echo "1. Install Chrome/Chromium browser"
        echo "2. Download ChromeDriver from https://chromedriver.chromium.org/"
        echo "3. Add ChromeDriver to your PATH"
        echo "4. Install Python dependencies: pip install weasyprint jinja2 selenium"
        return 0
    else
        echo "❌ Unsupported operating system: $OS"
        exit 1
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo "🐍 Installing Python dependencies..."
    
    # Check if uv is available
    if command -v uv &> /dev/null; then
        echo "📦 Using uv for dependency management..."
        uv sync
    else
        echo "❌ uv is required for dependency management. Please install uv first:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    # Check Python dependencies
    python3 -c "
import weasyprint
import jinja2
import selenium
from PIL import Image
print('✅ Python dependencies installed successfully')
" 2>/dev/null || echo "❌ Some Python dependencies are missing"
    
    # Check system dependencies
    if [[ "$OS" == "linux" ]]; then
        if command -v chromium &> /dev/null || command -v google-chrome &> /dev/null; then
            echo "✅ Chrome/Chromium browser found"
        else
            echo "❌ Chrome/Chromium browser not found"
        fi
        
        if command -v chromedriver &> /dev/null; then
            echo "✅ ChromeDriver found"
        else
            echo "❌ ChromeDriver not found"
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v chromedriver &> /dev/null; then
            echo "✅ ChromeDriver found"
        else
            echo "❌ ChromeDriver not found"
        fi
    fi
}

# Function to create pyproject.toml if it doesn't exist
create_pyproject() {
    if [[ ! -f "pyproject.toml" ]]; then
        echo "📝 Creating pyproject.toml..."
        cat > pyproject.toml << EOF
[project]
name = "accraiq"
version = "0.1.0"
description = "AccraIQ Transit Optimization Dashboard"
requires-python = ">=3.8"
dependencies = [
    "weasyprint>=59.0",
    "jinja2>=3.1.0",
    "selenium>=4.0.0",
    "Pillow>=9.0.0",
    "webdriver-manager>=4.0.0",
    "dtw-python>=1.5.3",
    "folium>=0.20.0",
    "geopandas>=1.1.1",
    "h3>=4.3.0",
    "hdbscan>=0.8.40",
    "numpy>=2.3.1",
    "pandas>=2.3.1",
    "plotly>=6.2.0",
    "pulp>=3.2.1",
    "pyproj>=3.7.1",
    "scikit-learn>=1.7.0",
    "scipy>=1.16.0",
    "shapely>=2.1.1",
    "streamlit>=1.46.1",
    "streamlit-folium>=0.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF
    fi
}

# Main installation process
main() {
    echo "🎯 Starting AccraIQ dependency installation..."
    
    # Create pyproject.toml if needed
    create_pyproject
    
    # Install system dependencies
    install_system_deps
    
    # Install Python dependencies
    install_python_deps
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "🚀 To run AccraIQ:"
    echo "   streamlit run app/main.py"
    echo ""
    echo "📄 For PDF generation, ensure Chrome/Chromium and ChromeDriver are properly installed."
    echo ""
}

# Run main function
main "$@" 