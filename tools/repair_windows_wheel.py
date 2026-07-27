"""Vendor the OpenBLAS DLL into a Windows wheel.

Windows has no system BLAS, so the build links the one SciPy ships as a
wheel, and the resulting extension has a dependency on a DLL that lives in
another package's directory.  delvewheel copies it in and rewrites the
import table, but only if it can find it -- and it is not on PATH, it is
inside site-packages.

Hence this rather than a `delvewheel repair` in pyproject.toml: the library
directory is version- and environment-specific, so it has to be asked for at
repair time, and doing that inside a cmd.exe one-liner means quoting a
subshell into a TOML string.

Usage: python tools/repair_windows_wheel.py <dest_dir> <wheel>

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import subprocess
import sys


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    dest_dir, wheel = argv

    import scipy_openblas64

    library_dir = scipy_openblas64.get_lib_dir()

    command = [
        sys.executable, "-m", "delvewheel", "repair",
        "--add-path", library_dir,
        "-w", dest_dir,
        wheel,
    ]
    print("+ " + " ".join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
