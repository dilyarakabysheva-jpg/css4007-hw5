import sys
from pathlib import Path

# Make "import helpdesk" work when pytest is run from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
