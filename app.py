import os
import sys
import importlib

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the dashboard and reload it to force re-execution on Streamlit reruns
import gdoc_opt.dashboard
importlib.reload(gdoc_opt.dashboard)
