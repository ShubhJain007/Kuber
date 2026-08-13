import os
import sys

# make the repo root importable (so `import kuber` works no matter where pytest is run)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
