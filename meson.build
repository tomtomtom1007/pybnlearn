project(
  'pybnlearn',
  'c',
  version: '0.1.0.dev0',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['c_std=c11', 'buildtype=release', 'warning_level=0'],
)

py = import('python').find_installation(pure: false)

# An implicit declaration is assumed to return int, so on a 64-bit target the
# pointer a function actually returned is truncated to 32 bits and silently
# stored.  That is not a style problem: it is how the first Linux run
# segfaulted, from a strdup() that glibc hides in strict ANSI mode while
# macOS declares regardless.  Newer compilers reject it, older ones only
# warn, and the difference between those two is the difference between a
# build failure and a corrupted pointer -- so make it an error everywhere
# rather than wait for the toolchain to decide.
cc = meson.get_compiler('c')
if cc.has_argument('-Werror=implicit-function-declaration')
  add_project_arguments('-Werror=implicit-function-declaration', language: 'c')
endif

# And stop glibc hiding the POSIX surface in the first place.  c_std=c11 puts
# it in strict ANSI mode, where strdup, strcasecmp and friends are simply not
# declared -- which is what produced the truncation above.  _DEFAULT_SOURCE is
# what gcc itself defines at -std=gnu11; it only adds declarations, and it is
# glibc-specific, so it is inert on macOS and Windows.  Belt and braces with
# the flag above: this prevents the class, that one catches whatever slips
# through.
if host_machine.system() == 'linux'
  add_project_arguments('-D_DEFAULT_SOURCE', language: 'c')
endif

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
  # OpenBLAS provides both in one library, under either of these names; the
  # second is what the scipy-openblas wheels install, which is how the Windows
  # build gets a BLAS at all.
  lapack = dependency('openblas', required: false)
  blas = declare_dependency()
endif
if not lapack.found()
  lapack = dependency('scipy-openblas', required: false)
  blas = declare_dependency()
endif

# Windows has no system BLAS, so it arrives as a Python package instead.
# Asking the interpreter where it landed is more robust than writing a .pc
# file and pointing PKG_CONFIG_PATH at it: pkg-config runs with the build
# directory as its working directory, so a relative PKG_CONFIG_PATH -- which
# is the obvious thing to write -- silently finds nothing, and meson then
# falls through to a CMake probe that reports an unrelated failure about
# Visual Studio generators.
if not lapack.found()
  probe = run_command(py, '-c',
    'import scipy_openblas64 as ob;' +
    'print(ob.get_include_dir());' +
    'print(ob.get_lib_dir());' +
    'print(ob.get_library())',
    check: false)

  if probe.returncode() == 0
    parts = probe.stdout().strip().split('\n')
    ob_include = parts[0].strip()
    ob_libdir = parts[1].strip()
    ob_library = parts[2].strip()

    ob_lib = meson.get_compiler('c').find_library(
      ob_library, dirs: [ob_libdir], required: false)

    if ob_lib.found()
      lapack = declare_dependency(
        include_directories: include_directories(ob_include),
        dependencies: [ob_lib])
      blas = declare_dependency()
    endif
  endif
endif

if not lapack.found()
  if host_machine.system() == 'darwin'
    lapack = declare_dependency(link_args: ['-framework', 'Accelerate'])
    blas = declare_dependency()
  else
    error('no BLAS/LAPACK found; install one (libopenblas-dev, openblas-devel, '
          + 'or the scipy-openblas64 wheel) and retry')
  endif
endif

m_dep = meson.get_compiler('c').find_library('m', required: false)

subdir('src/c')
subdir('src/pybnlearn')
