cython = find_program('cython')

core_c = custom_target(
  '_core.c',
  input: '_core.pyx',
  output: '_core.c',
  command: [cython, '-3', '--module-name', 'pybnlearn._core',
            '@INPUT@', '-o', '@OUTPUT@'],
)

py.extension_module(
  '_core',
  core_c,
  # R_NO_REMAP turns off rcompat.h's unprefixed aliases for this translation
  # unit only.  It is the one that also includes Python.h, and the aliases are
  # common words: `length` collides with PyASCIIObject's member of that name,
  # rewriting `obj->length` into `obj->Rf_length`.  The binding calls Rf_*
  # directly, so it loses nothing; the vendored bnlearn sources still get the
  # aliases, because they are written against them.
  c_args: np_inc_args + ['-DR_NO_REMAP'],
  dependencies: [pybn_core_dep, py.dependency()],
  subdir: 'pybnlearn',
  install: true,
)

py.install_sources(
  ['__init__.py', 'bootstrap.py', 'causal.py', 'classifiers.py',
   'constraint.py', 'divergence.py', 'exact.py', 'fit.py', 'foreign.py',
   'graph.py', 'hybrid.py', 'inference.py', 'lingam.py', 'missing.py',
   'mvnorm.py',
   'nodes.py', 'preprocessing.py', 'strength.py', 'structure.py'],
  subdir: 'pybnlearn',
)
