import os
import sys

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the dashboard
if __name__ == "__main__":
    import streamlit as st
    # Import the dashboard module to execute it
    import gdoc_opt.dashboard
