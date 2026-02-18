"""
Path utilities for cross-platform compatibility.
"""
from .paths import (
    get_project_root,
    get_weights_path,
    get_data_path,
    get_output_path,
    get_src_path,
    setup_kaggle_paths,
    get_paths
)

__all__ = [
    'get_project_root',
    'get_weights_path',
    'get_data_path',
    'get_output_path',
    'get_src_path',
    'setup_kaggle_paths',
    'get_paths'
]
