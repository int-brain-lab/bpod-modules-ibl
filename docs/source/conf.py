import importlib.util
import sys
from datetime import date
from pathlib import Path

project_root = Path(__file__).parents[2].resolve()
docs_source_path = Path(__file__).parent.resolve()
sys.path.insert(0, project_root)
sys.path.insert(0, str(docs_source_path / "_ext"))

__version__ = importlib.metadata.version("bpod_modules_ibl")


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bpod-modules-ibl"
copyright = f"{date.today().year}, International Brain Laboratory"  # noqa: A001
author = "International Brain Laboratory"
release = ".".join(__version__.split(".")[:3])
version = ".".join(__version__.split(".")[:3])
rst_prolog = f"""
.. |version_code| replace:: ``{version}``
"""

html_context = {
    "display_github": False,
    "github_user": "int-brain-lab",
    "github_repo": "bpod-modules-ibl",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "sphinx_github_style",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx-jsonschema",
    "sphinx_toolbox.wikipedia",
    "doctest_codeblock",
]
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_timeout = 30
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "serial": ("https://pyserial.readthedocs.io/en/stable", None),
    "bpod-core": ("https://int-brain-lab.github.io/bpod-core/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Settings for automatic API generation -----------------------------------
autodoc_mock_imports = ["_typeshed"]
autodoc_class_signature = "separated"  # 'mixed', 'separated'
autodoc_member_order = "groupwise"  # 'alphabetical', 'groupwise', 'bysource'
autodoc_inherit_docstrings = False
autodoc_typehints = "description"  # 'description', 'signature', 'none', 'both'
autodoc_typehints_description_target = "all"  # 'all', 'documented', 'documented_params'
autodoc_typehints_format = "short"  # 'fully-qualified', 'short'

autosummary_generate = True
autosummary_imported_members = False

typehints_defaults = None
typehints_use_rtype = True
typehints_use_signature = False
typehints_use_signature_return = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "DataFrame": "pandas.DataFrame",
    "Series": "pandas.Series",
    "Mapping": "collections.abc.Mapping",
    "ValidationError": "pydantic.ValidationError",
}
napoleon_attr_annotations = True

graphviz_output_format = "svg"
graphviz_inline = False

numfig = True
html_static_path = ["_static"]
html_css_files = ["custom.css"]

linkcode_link_text = " "
pygments_style = "default"
highlight_language = "python3"
