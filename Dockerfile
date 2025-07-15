# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for geospatial libraries and PDF generation
RUN apt-get update && apt-get install -y \
    curl \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libspatialindex-dev \
    # WeasyPrint dependencies
    python3-dev \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    # Additional WeasyPrint dependencies
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    # Chrome/Chromium for Selenium map screenshots
    chromium \
    chromium-driver \
    # Additional system utilities
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome/Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the entire project for uv sync
COPY . .

# Install dependencies using uv sync
RUN uv sync --frozen --no-cache

# Create data directory if it doesn't exist
RUN mkdir -p data



# Expose the port (Streamlit default is 8501)
EXPOSE 8501

# Run the application
CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]