import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "pysinger"
copyright = "2025"
author = "Kevin Korfmann"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"

autodoc_mock_imports = ["numpy", "matplotlib", "tskit", "tqdm", "sortedcontainers"]
