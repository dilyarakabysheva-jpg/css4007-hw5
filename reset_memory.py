"""Delete the saved memory (conversation checkpoints and user preferences).

    python reset_memory.py
"""

import shutil

from helpdesk.session import storage_dir

folder = storage_dir()
shutil.rmtree(folder)
print(f"Deleted {folder} (conversation memory and user memory).")
