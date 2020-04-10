import os
import sys

sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

author = 'Mike S Wang'
copyright = '2020, M S Wang'
project = 'Harmonia'
release = '0.1'


# -- General configuration ---------------------------------------------------

source_suffix = ['.rst', '.txt', '.md']

master_doc = 'harmonia'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']

pygments_style = 'sphinx'

language = None

exclude_patterns = ['tests']


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    'page_width': '1150px',
    'sidebar_width': '250px',
    'fixed_sidebar' : True,
    'github_user': 'MikeSWang',
    'github_repo': 'Harmonia',
}

html_static_path = ['_static']

html_logo = '_static/Harmonia.png'

html_sidebars = {
    '**': ['navigation.html', 'searchbox.html'],
    'using/windows': ['windowssidebar.html', 'searchbox.html'],
}

htmlhelp_basename = 'Harmonia_doc'


# -- Extension configuration -------------------------------------------------

autodoc_member_order = 'bysource'
autosummary_generate = True

napoleon_include_special_with_doc = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ("https://docs.python.org/3", None),
    'pytest': ("http://docs.pytest.org/en/latest/", None),
    'numpy': ("https://docs.scipy.org/doc/numpy", None),
    'scipy': ("https://docs.scipy.org/doc/scipy/reference", None),
    'matplotlib': ("https://matplotlib.org", None),
    'mpi4py': ("https://mpi4py.readthedocs.io/en/latest", None),
    'mpmath': ("http://mpmath.org/doc/1.1.0/", None),
    'nbodykit': ("https://nbodykit.readthedocs.io/en/latest", None),
}

todo_include_todos = True
