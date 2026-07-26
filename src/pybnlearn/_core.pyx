# cython: language_level=3
"""Conversion between Python objects and the R objects the C core expects.

The vendored bnlearn entry points take and return SEXPs, so this module is the
whole of the binding layer: it builds SEXPs out of pandas and NumPy objects,
calls an entry point under `pybn_protected_call`, and converts the result back.

Two invariants matter here.

* Everything the C core allocates lives in an arena that is freed when the call
  returns, so results have to be *copied* into Python objects before
  `pybn_arena_pop`.  Nothing may hold a SEXP past the end of a call.
* `error()` in the C core unwinds with `longjmp`, which must not cross code
  Cython generated.  It is caught in C (see compat/guarded_call.c); this module
  only ever sees a return code.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import warnings

import numpy as np
import pandas as pd

cimport numpy as cnp

cnp.import_array()


cdef extern from "rcompat.h":

    ctypedef struct SEXPREC:
        pass
    ctypedef SEXPREC *SEXP

    cdef enum:
        NILSXP, SYMSXP, LISTSXP, CLOSXP, LANGSXP, CHARSXP
        LGLSXP, INTSXP, REALSXP, STRSXP, VECSXP

    SEXP R_NilValue
    SEXP R_NamesSymbol
    SEXP R_ClassSymbol
    SEXP R_LevelsSymbol
    SEXP R_RowNamesSymbol

    int NA_INTEGER

    int TYPEOF(SEXP x)
    int *INTEGER(SEXP x)
    int *LOGICAL(SEXP x)
    double *REAL(SEXP x)
    const char *CHAR(SEXP x)

    SEXP Rf_allocVector(int type, int n)
    int Rf_length(SEXP x)
    SEXP Rf_STRING_ELT(SEXP x, int i)
    void Rf_SET_STRING_ELT(SEXP x, int i, SEXP v)
    SEXP Rf_VECTOR_ELT(SEXP x, int i)
    void Rf_SET_VECTOR_ELT(SEXP x, int i, SEXP v)
    SEXP Rf_mkChar(const char *s)
    SEXP Rf_mkString(const char *s)
    SEXP Rf_ScalarReal(double x)
    SEXP Rf_ScalarInteger(int x)
    SEXP Rf_ScalarLogical(int x)
    SEXP Rf_getAttrib(SEXP x, SEXP name)
    SEXP Rf_setAttrib(SEXP x, SEXP name, SEXP value)


cdef extern from "rcompat_internal.h":

    void pybn_init_constants()
    void pybn_arena_push()
    void pybn_arena_pop()
    const char *pybn_error_message()
    int pybn_warning_count()
    const char *pybn_warning_at(int i)
    void pybn_clear_warnings()
    const char *pybn_output_buffer()
    void pybn_clear_output()


cdef extern int pybn_protected_call(void *fn, SEXP *args, int nargs, SEXP *out)

# The bnlearn entry points reached so far.  Declaring them rather than looking
# them up by name means the linker checks the arity.
cdef extern SEXP onLoad()
cdef extern SEXP indep_test(SEXP x, SEXP y, SEXP sx, SEXP data, SEXP test,
    SEXP alpha, SEXP extra_args, SEXP learning, SEXP complete)


class BNLearnError(RuntimeError):
    """Raised when the C core calls error()."""


cdef bint _initialised = False


cdef void _ensure_init() noexcept:
    """onLoad() interns the symbols bnlearn caches in globals; it has to run
    once before any entry point is called."""
    if not _initialised:
        pybn_init_constants()
        onLoad()
        _set_initialised()


cdef void _set_initialised() noexcept:
    global _initialised
    _initialised = True


cdef class _Raw:
    """Carries an already-built SEXP through an argument list."""
    cdef SEXP ptr


cdef _Raw _raw(SEXP p):
    cdef _Raw r = _Raw.__new__(_Raw)
    r.ptr = p
    return r


# ---------------------------------------------------------------------------
# Python -> SEXP
# ---------------------------------------------------------------------------

cdef SEXP _str_vector(values):
    """A character vector."""
    cdef SEXP out = Rf_allocVector(STRSXP, len(values))
    cdef bytes encoded
    cdef int i
    for i, v in enumerate(values):
        encoded = str(v).encode("utf-8")
        Rf_SET_STRING_ELT(out, i, Rf_mkChar(encoded))
    return out


cdef SEXP _named_logical(names, bint value):
    """A logical vector carrying a names attribute, as bnlearn's `complete`
    argument does."""
    cdef SEXP out = Rf_allocVector(LGLSXP, len(names))
    cdef int i
    for i in range(len(names)):
        LOGICAL(out)[i] = 1 if value else 0
    Rf_setAttrib(out, R_NamesSymbol, _str_vector(names))
    return out


cdef SEXP _factor_column(codes, categories):
    """A factor: one-based integer codes plus a levels attribute.  pandas marks
    missing values with -1 where R uses NA_INTEGER, so the codes are shifted."""
    cdef cnp.ndarray[cnp.int64_t, ndim=1] c = np.ascontiguousarray(
        codes, dtype=np.int64)
    cdef int n = c.shape[0]
    cdef SEXP out = Rf_allocVector(INTSXP, n)
    cdef int i
    for i in range(n):
        INTEGER(out)[i] = NA_INTEGER if c[i] < 0 else <int>(c[i] + 1)
    Rf_setAttrib(out, R_LevelsSymbol, _str_vector(categories))
    Rf_setAttrib(out, R_ClassSymbol, Rf_mkString(b"factor"))
    return out


cdef SEXP _numeric_column(values):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] v = np.ascontiguousarray(
        values, dtype=np.float64)
    cdef int n = v.shape[0]
    cdef SEXP out = Rf_allocVector(REALSXP, n)
    cdef int i
    for i in range(n):
        REAL(out)[i] = v[i]
    return out


cdef SEXP _dataframe(object df):
    """Build the data.frame SEXP bnlearn expects.

    Categorical columns become R factors and numeric columns stay numeric.
    bnlearn decides whether a network is discrete, Gaussian or conditional
    Gaussian purely from these column types, so the mapping has to be exact.
    """
    cdef int ncol = df.shape[1]
    cdef int nrow = df.shape[0]
    cdef SEXP out = Rf_allocVector(VECSXP, ncol)
    cdef SEXP rownames = Rf_allocVector(INTSXP, nrow)
    cdef SEXP col
    cdef int i, j

    for j, name in enumerate(df.columns):
        series = df[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            col = _factor_column(series.cat.codes.to_numpy(),
                                 list(series.cat.categories))
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            cat = series.astype("category")
            col = _factor_column(cat.cat.codes.to_numpy(),
                                 list(cat.cat.categories))
        elif pd.api.types.is_bool_dtype(series):
            cat = series.astype("category")
            col = _factor_column(cat.cat.codes.to_numpy(),
                                 [str(v) for v in cat.cat.categories])
        elif pd.api.types.is_numeric_dtype(series):
            col = _numeric_column(series.to_numpy())
        else:
            raise TypeError(
                f"column {name!r} has dtype {series.dtype}, which is neither "
                "categorical nor numeric")
        Rf_SET_VECTOR_ELT(out, j, col)

    for i in range(nrow):
        INTEGER(rownames)[i] = i + 1

    Rf_setAttrib(out, R_NamesSymbol, _str_vector([str(c) for c in df.columns]))
    Rf_setAttrib(out, R_ClassSymbol, Rf_mkString(b"data.frame"))
    Rf_setAttrib(out, R_RowNamesSymbol, rownames)

    return out


cdef SEXP _py_to_sexp(object obj):
    """Convert the scalars and containers that make up bnlearn's arguments."""
    cdef SEXP out
    cdef bytes encoded
    cdef int i

    if obj is None:
        return R_NilValue
    if isinstance(obj, _Raw):
        return (<_Raw>obj).ptr
    if isinstance(obj, str):
        encoded = obj.encode("utf-8")
        return Rf_mkString(encoded)
    if isinstance(obj, (bool, np.bool_)):
        return Rf_ScalarLogical(1 if obj else 0)
    if isinstance(obj, (int, np.integer)):
        return Rf_ScalarInteger(<int>obj)
    if isinstance(obj, (float, np.floating)):
        return Rf_ScalarReal(<double>obj)
    if isinstance(obj, dict):
        out = Rf_allocVector(VECSXP, len(obj))
        for i, (k, v) in enumerate(obj.items()):
            Rf_SET_VECTOR_ELT(out, i, _py_to_sexp(v))
        Rf_setAttrib(out, R_NamesSymbol,
                     _str_vector([str(k) for k in obj.keys()]))
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) and all(isinstance(v, str) for v in obj):
            return _str_vector(obj)
        return _numeric_column(np.asarray(obj, dtype=np.float64))
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "US":
            return _str_vector(obj.tolist())
        return _numeric_column(obj)

    raise TypeError(f"cannot convert {type(obj).__name__} to an R object")


# ---------------------------------------------------------------------------
# SEXP -> Python
# ---------------------------------------------------------------------------

cdef object _names_of(SEXP x):
    cdef SEXP nm = Rf_getAttrib(x, R_NamesSymbol)
    cdef int i
    if nm == R_NilValue:
        return None
    return [CHAR(Rf_STRING_ELT(nm, i)).decode("utf-8")
            for i in range(Rf_length(nm))]


cdef object _sexp_to_py(SEXP x):
    """Copy a SEXP into Python objects.

    Everything is copied rather than viewed: the arena that owns `x` is freed
    as soon as the call returns.
    """
    cdef int n, i, t
    cdef SEXP levels
    cdef cnp.ndarray arr

    if x is NULL or x == R_NilValue:
        return None

    t = TYPEOF(x)
    n = Rf_length(x)
    names = _names_of(x)

    if t == VECSXP:
        values = [_sexp_to_py(Rf_VECTOR_ELT(x, i)) for i in range(n)]
        return dict(zip(names, values)) if names else values

    if t == LGLSXP:
        out = [None if LOGICAL(x)[i] == NA_INTEGER else bool(LOGICAL(x)[i])
               for i in range(n)]
    elif t == INTSXP:
        levels = Rf_getAttrib(x, R_LevelsSymbol)
        if levels != R_NilValue:
            cats = [CHAR(Rf_STRING_ELT(levels, i)).decode("utf-8")
                    for i in range(Rf_length(levels))]
            arr = np.empty(n, dtype=np.int64)
            for i in range(n):
                arr[i] = -1 if INTEGER(x)[i] == NA_INTEGER \
                    else INTEGER(x)[i] - 1
            return pd.Categorical.from_codes(arr, categories=cats)
        arr = np.empty(n, dtype=np.int64)
        for i in range(n):
            arr[i] = INTEGER(x)[i]
        out = arr
    elif t == REALSXP:
        arr = np.empty(n, dtype=np.float64)
        for i in range(n):
            arr[i] = REAL(x)[i]
        out = arr
    elif t == STRSXP:
        out = [CHAR(Rf_STRING_ELT(x, i)).decode("utf-8") for i in range(n)]
    else:
        raise TypeError(f"cannot convert an R object of type {t} to Python")

    if names is not None:
        return dict(zip(names, list(out)))

    # an unnamed length-one vector is a scalar to Python eyes.
    if n == 1:
        return out[0]

    return out


# ---------------------------------------------------------------------------
# calling an entry point
# ---------------------------------------------------------------------------

cdef object _call(void *fn, list args, bint capture_output=False):
    """Run an entry point with the arena and the error handler in place.

    The caller is responsible for the arena, because arguments built by the
    caller (a data.frame, say) have to outlive the conversion done here.
    """
    cdef SEXP argv[12]
    cdef SEXP result = NULL
    cdef int rc, i
    cdef int nargs = len(args)

    if nargs > 12:
        raise ValueError("entry points take at most 12 arguments")

    pybn_clear_warnings()
    pybn_clear_output()

    for i in range(nargs):
        argv[i] = _py_to_sexp(args[i])

    rc = pybn_protected_call(fn, argv, nargs, &result)

    if rc != 0:
        raise BNLearnError(pybn_error_message().decode("utf-8"))

    value = _sexp_to_py(result)

    for i in range(pybn_warning_count()):
        warnings.warn(pybn_warning_at(i).decode("utf-8"), stacklevel=3)

    if capture_output:
        return value, pybn_output_buffer().decode("utf-8")

    return value


# ---------------------------------------------------------------------------
# the public entry points
# ---------------------------------------------------------------------------

def ci_test(data, x, y, sx=None, test="mi", alpha=1.0, extra_args=None):
    """Run a conditional independence test, mirroring bnlearn's ci.test().

    Parameters
    ----------
    data : pandas.DataFrame
        Categorical columns are treated as discrete, numeric ones as Gaussian.
    x, y : str
        The two variables to test.
    sx : sequence of str, optional
        The conditioning set.
    test : str
        A bnlearn test label: "mi", "x2", "mi-adf", "cor", "zf", and so on.
    alpha : float
    extra_args : dict, optional
        Test-specific arguments, e.g. {"B": 100} for permutation tests.

    Returns
    -------
    dict
        The fields of R's htest object: statistic, parameter, p.value.
    """
    cdef SEXP data_sexp
    cdef SEXP complete_sexp

    _ensure_init()
    pybn_arena_push()

    try:
        data_sexp = _dataframe(data)
        complete_sexp = _named_logical([str(c) for c in data.columns], True)

        return _call(<void *>indep_test, [
            x,
            y,
            list(sx) if sx else None,
            _raw(data_sexp),
            test,
            float(alpha),
            extra_args or {},
            False,
            _raw(complete_sexp),
        ])
    finally:
        pybn_arena_pop()
