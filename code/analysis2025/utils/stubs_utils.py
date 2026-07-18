"""Disk cache helpers for expensive per-video computations."""

import os
import pickle

def save_stub(stub_path,object):
    """Pickles an object to disk, creating the parent directory if needed."""
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))

    if stub_path is not None:
        with open(stub_path,'wb') as f:
            pickle.dump(object,f)

def read_stub(read_from_stub,stub_path):
    """Loads a pickled object from disk if caching is enabled and the file exists."""
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path,'rb') as f:
            object = pickle.load(f)
            return object
    return None
