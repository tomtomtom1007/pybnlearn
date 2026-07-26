project(
  'pybnlearn',
  'c', 'fortran',
  version: '0.1.0.dev0',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['c_std=c11', 'buildtype=release', 'warning_level=0'],
)

py = import('python').find_installation(pure: false)

incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

# Passed as a compiler flag rather than through include_directories(): meson
# rejects absolute paths that land inside the source tree, which is exactly
# what happens when the build runs from a virtualenv created in the project
# directory.  A -I flag has no such restriction and does not care where the
# environment lives.
np_inc_args = ['-I' + incdir_numpy]

# BLAS and LAPACK.  bnlearn reaches for exactly four routines (dgemm, dgesdd,
# dgetrf, and the LINPACK QR below), so any implementation will do; SciPy has
# already put one on the system wherever pybnlearn can be installed.
lapack = dependency('lapack', required: false)
blas = dependency('blas', required: false)
if not lapack.found()
  if host_machine.system() == 'darwin'
    lapack = declare_dependency(link_args: ['-framework', 'Accelerate'])
    blas = declare_dependency()
  else
    error('no LAPACK found; install one (e.g. libopenblas-dev) and retry')
  endif
endif

m_dep = meson.get_compiler('c').find_library('m', required: false)

subdir('src/c')
subdir('src/pybnlearn')
