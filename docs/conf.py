# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

# load extensions
extensions = ['autoapi.extension', 'myst_nb', 'sphinx.ext.intersphinx']

# specify project details
master_doc = 'index'
project = 'NiAS'
author = 'NiAS Developers and Contributors'
copyright = 'NiAS Developers and Contributors'
version = subprocess.run(['hatch', 'version'], capture_output=True).stdout.decode()

# basic build settings
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
nitpicky = True

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
}

import substitutions  # noqa: I001
rst_epilog = substitutions.substitutions

suppress_warnings = ['autoapi']

autoapi_dirs = ['../src']
autoapi_type = 'python'
autoapi_keep_files = False
suppress_warnings = ['autoapi']
autoapi_member_order = 'groupwise'
autoapi_options = [
    'show-inheritance',
    'show-module-summary',
    'members',
    'undoc-members'
]

myst_enable_extensions = [
    'amsmath',
    'deflist',
    'dollarmath',
    'smartquotes',
    'strikethrough',
    'substitution',
]
myst_dmath_double_inline = True  # allow $$ .. $$ without surrounding blank lines
myst_heading_anchors = 3  # automatically adds 'title-of-heading' anchors for headings
myst_substitutions = substitutions.myst_substitutions

## myst_nb default settings

# Custom formats for reading notebook; suffix -> reader
# nb_custom_formats = {}

# Notebook level metadata key for config overrides
# nb_metadata_key = 'mystnb'

# Cell level metadata key for config overrides
# nb_cell_metadata_key = 'mystnb'

# Mapping of kernel name regex to replacement kernel name(applied before execution)
# nb_kernel_rgx_aliases = {}

# Regex that matches permitted values of eval expressions
# nb_eval_name_regex = '^[a-zA-Z_][a-zA-Z0-9_]*$'

# Execution mode for notebooks
# nb_execution_mode = 'auto'

# Path to folder for caching notebooks (default: <outdir>)
# nb_execution_cache_path = ''

# Exclude (POSIX) glob patterns for notebooks
# nb_execution_excludepatterns = ()

# Execution timeout (seconds)
# nb_execution_timeout = 30

# Use temporary folder for the execution current working directory
# nb_execution_in_temp = False

# Allow errors during execution
# nb_execution_allow_errors = False

# Raise an exception on failed execution, rather than emitting a warning
# nb_execution_raise_on_error = False

# Print traceback to stderr on execution error
# nb_execution_show_tb = False

# Merge stdout/stderr execution output streams
# nb_merge_streams = False

# The entry point for the execution output render class (in group `myst_nb.output_renderer`)
# nb_render_plugin = 'default'

# Remove code cell source
# nb_remove_code_source = False

# Remove code cell outputs
# nb_remove_code_outputs = False

# Prompt to expand hidden code cell {content|source|outputs}
# nb_code_prompt_show = 'Show code cell {type}'

# Prompt to collapse hidden code cell {content|source|outputs}
# nb_code_prompt_hide = 'Hide code cell {type}'

# Number code cell source lines
# nb_number_source_lines = False

# Overrides for the base render priority of mime types: list of (builder name, mime type, priority)
# nb_mime_priority_overrides = ()

# Behaviour for stderr output
# nb_output_stderr = 'show'

# Pygments lexer applied to stdout/stderr and text/plain outputs
# nb_render_text_lexer = 'myst-ansi'

# Pygments lexer applied to error/traceback outputs
# nb_render_error_lexer = 'ipythontb'

# Options for image outputs (class|alt|height|width|scale|align)
# nb_render_image_options = {}

# Options for figure outputs (classes|name|caption|caption_before)
# nb_render_figure_options = {}

# The format to use for text/markdown rendering
# nb_render_markdown_format = 'commonmark'


html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/nias-project/nias',
    'use_repository_button': True,
    'show_toc_level': 1,
}
html_title = 'NiAS<br>Numerics in Abstract Spaces'
html_static_path = ['_static']
html_css_files = ['custom.css']


TYPE_ALIASES = ['Indices', 'NDArray', 'ArrayLike', 'Scalar']


def resolve_type_aliases(app, env, node, contnode):
    if (node['refdomain'] == 'py'
        and node['reftype'] == 'class'
        and node['reftarget'].split('.')[-1] in TYPE_ALIASES):

        from sphinx.errors import NoUri
        raise NoUri


def setup(app):
    app.connect('missing-reference', resolve_type_aliases)
