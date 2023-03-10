# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../lvgp_pytorch/'))

# - Copy over examples folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images

examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "notebooks"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "notebooks"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)
            shutil.copyfile(source_filename, dest_filename)


# -- Project information -----------------------------------------------------

project = 'lvgp_pytorch'
copyright = '2021, Northwestern University'
author = 'Suraj Yerramilli, Wei Chen, Daniel W. Apley, Ramin Bostanabad'

# The full version, including alpha/beta/rc tags
release = '0.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "sphinx.ext.mathjax",
    "nbsphinx"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

master_doc = 'contents'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'\usepackage{enumitem}\setlistdepth{99}',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "lvgp-pytorch-manual.tex", "LVGP-PyTorch User Manual", "Northwestern University", "manual")
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "lvgp-pytorch", "LVGP-PyTorch User Manual", [author], 1)]