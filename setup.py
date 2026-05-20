from setuptools import setup, find_packages

setup(
    name="gdoc_opt",
    version="1.0.0",
    description="Global Distribution Center Optimization System",
    author="Nishant Iyer",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.0.0",
        "geopy>=2.3.0",
        "pulp>=2.7.0",
        "torch>=2.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "streamlit>=1.22.0",
        "plotly>=5.14.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.20.0",
    ],
    python_requires=">=3.9",
)
