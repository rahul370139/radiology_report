#!/usr/bin/env python3
"""
Patch for Accelerator compatibility with transformers 4.36.2
Fixes the use_seedable_sampler parameter issue
"""

import sys
from accelerate import Accelerator

# Store original __init__ method
_original_init = Accelerator.__init__

def patched_init(self, *args, **kwargs):
    """Patched __init__ that removes unsupported parameters"""
    # Remove unsupported parameters
    kwargs.pop('use_seedable_sampler', None)
    kwargs.pop('seed_worker', None)
    
    # Call original init with cleaned kwargs
    return _original_init(self, *args, **kwargs)

# Apply the patch
Accelerator.__init__ = patched_init

print("✅ Accelerator compatibility patch applied")
