import os
import sys

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Remove from sys.modules cache to force fresh import on rerun
if "gdoc_opt.dashboard" in sys.modules:
    del sys.modules["gdoc_opt.dashboard"]

# Import the dashboard to execute it
import gdoc_opt.dashboard
