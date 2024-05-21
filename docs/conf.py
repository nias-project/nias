# outline for a myst_nb project with sphinx
# build with: sphinx-build -nW --keep-going -b html . ./_build/html
import subprocess

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

html_theme = 'sphinx_material'
html_theme_options = {
    'nav_title': 'NiAS Documentation',
    'globaltoc_depth': 6,
    'theme_color': 'indigo',
    'color_primary': 'indigo',
    'color_accent': 'blue',
    'repo_url': 'https://github.com/nias-project/nias',
    'repo_name': 'nias',
    'logo_icon': '&#xe0e0',
}
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f'{project} v{version} Manual'

# The name of an image file (within the static path) to place at the top of
# the sidebar.
# html_logo = 'TODO'

# The name of an image file to use as favicon.
# html_favicon = 'TODO'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# all: "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']
}
# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {
#    'index': 'indexcontent.html',
# }

# html_css_files = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css']

# If false, no module index is generated.
html_use_modindex = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".html").
# html_file_suffix = '.html'

# Hide link to page source.
html_show_sourcelink = False


TYPE_ALIASES = ['Indices', 'NDArray', 'ArrayLike', 'Scalar']


def resolve_type_aliases(app, env, node, contnode):
    if (node['refdomain'] == 'py'
        and node['reftype'] == 'class'
        and node['reftarget'].split('.')[-1] in TYPE_ALIASES):

        from sphinx.errors import NoUri
        raise NoUri


def setup(app):
    app.connect('missing-reference', resolve_type_aliases)
