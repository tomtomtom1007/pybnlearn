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
  c_args: np_inc_args,
  dependencies: [pybn_core_dep, py.dependency()],
  subdir: 'pybnlearn',
  install: true,
)

py.install_sources(
  ['__init__.py', 'bootstrap.py', 'classifiers.py', 'constraint.py', 'exact.py', 'fit.py', 'graph.py', 'hybrid.py',
   'inference.py', 'mvnorm.py', 'nodes.py', 'strength.py',
   'structure.py'],
  subdir: 'pybnlearn',
)
