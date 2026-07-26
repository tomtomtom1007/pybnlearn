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
    SEXP R_DimSymbol

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
    SEXP Rf_install(const char *name)


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
    if isinstance(obj, pd.Series):
        # A named numeric vector.  Some scores index their arguments by node
        # name rather than by position -- BGe's prior mean vector is looked up
        # with subset_by_name() -- so the names have to travel with the values.
        out = _numeric_column(obj.to_numpy())
        Rf_setAttrib(out, R_NamesSymbol,
                     _str_vector([str(k) for k in obj.index]))
        return out
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
    cdef SEXP argv[16]
    cdef SEXP result = NULL
    cdef int rc, i
    cdef int nargs = len(args)

    if nargs > 16:
        raise ValueError("entry points take at most 16 arguments")

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




# ---------------------------------------------------------------------------
# structure learning
#
# The score-based search carries state between iterations, and two pieces of it
# are mutated by the C code rather than returned:
#
#   cache      score_cache_fill() writes the cached score deltas into it, so
#              that each iteration only rescores the nodes whose parents moved.
#   reference  hc_opt_step() adds the chosen operation's delta straight into
#              the per-node reference scores (hc.cache.lookup.c:352).
#
# Both therefore have to be the *same* SEXP from one call to the next, which is
# why they are allocated in the arena this object owns rather than rebuilt per
# call.  Each method wraps its own call in a nested arena so that per-iteration
# garbage is released instead of piling up in the outer one.
#
# Every entry point goes through _guarded(): the C core reports failure by
# calling error(), which longjmps, and without a setjmp in place that jump
# lands on a stale frame and takes the interpreter down instead of raising.
# ---------------------------------------------------------------------------

cdef extern SEXP arcs2amat(SEXP arcs, SEXP nodes)
cdef extern SEXP cache_structure(SEXP nodes, SEXP amat, SEXP debug)
cdef extern SEXP per_node_score(SEXP network, SEXP data, SEXP score,
    SEXP targets, SEXP extra, SEXP debug)
cdef extern SEXP score_cache_fill(SEXP nodes, SEXP data, SEXP network,
    SEXP score, SEXP extra, SEXP reference, SEXP equivalence,
    SEXP decomposability, SEXP updated, SEXP amat, SEXP cache, SEXP blmat,
    SEXP debug)
cdef extern SEXP hc_opt_step(SEXP amat, SEXP nodes, SEXP added, SEXP cache,
    SEXP reference, SEXP wlmat, SEXP blmat, SEXP nparents, SEXP maxp,
    SEXP debug)
cdef extern SEXP tabu_hash(SEXP amat, SEXP nodes, SEXP list, SEXP current)
cdef extern SEXP tabu_step(SEXP amat, SEXP nodes, SEXP added, SEXP cache,
    SEXP reference, SEXP wlmat, SEXP blmat, SEXP tabu_list, SEXP current,
    SEXP baseline, SEXP nparents, SEXP maxp, SEXP debug)
cdef extern SEXP hc_to_be_added(SEXP arcs, SEXP blacklist, SEXP whitelist,
    SEXP nparents, SEXP maxp, SEXP nodes, SEXP convert)
cdef extern SEXP is_acyclic(SEXP arcs, SEXP nodes, SEXP return_nodes,
    SEXP directed, SEXP debug)


cdef SEXP _guarded(void *fn, SEXP *args, int nargs) except? NULL:
    """Call an entry point with the error handler in place."""
    cdef SEXP out = NULL
    if pybn_protected_call(fn, args, nargs, &out) != 0:
        raise BNLearnError(pybn_error_message().decode("utf-8"))
    return out


cdef SEXP _int_vector(object values):
    """A plain integer vector, with no dim attribute."""
    cdef cnp.ndarray[cnp.int32_t, ndim=1] v = np.ascontiguousarray(
        np.asarray(values, dtype=np.int32).ravel())
    cdef int n = v.shape[0]
    cdef SEXP out = Rf_allocVector(INTSXP, n)
    cdef int i
    for i in range(n):
        INTEGER(out)[i] = v[i]
    return out


cdef SEXP _int_matrix(object arr):
    """An integer matrix in R's column-major layout."""
    cdef cnp.ndarray[cnp.int32_t, ndim=2] a = np.ascontiguousarray(
        arr, dtype=np.int32)
    cdef int nrow = a.shape[0], ncol = a.shape[1]
    cdef SEXP out = Rf_allocVector(INTSXP, nrow * ncol)
    cdef SEXP dim = Rf_allocVector(INTSXP, 2)
    cdef int i, j
    for j in range(ncol):
        for i in range(nrow):
            INTEGER(out)[j * nrow + i] = a[i, j]
    INTEGER(dim)[0] = nrow
    INTEGER(dim)[1] = ncol
    Rf_setAttrib(out, R_DimSymbol, dim)
    return out


cdef object _read_int_matrix(SEXP x, int nrow, int ncol):
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out = np.empty(
        (nrow, ncol), dtype=np.int32)
    cdef int i, j
    for j in range(ncol):
        for i in range(nrow):
            out[i, j] = INTEGER(x)[j * nrow + i]
    return out


cdef SEXP _real_named(object values, object names):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] v = np.ascontiguousarray(
        np.asarray(values, dtype=np.float64).ravel())
    cdef int n = v.shape[0]
    cdef SEXP out = Rf_allocVector(REALSXP, n)
    cdef int i
    for i in range(n):
        REAL(out)[i] = v[i]
    if names is not None:
        Rf_setAttrib(out, R_NamesSymbol, _str_vector(names))
    return out


cdef object _read_real(SEXP x, int n):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        out[i] = REAL(x)[i]
    return out


cdef SEXP _arcs_sexp(object arcs):
    """bnlearn hands arc sets to C as a flat character vector: the whole 'from'
    column followed by the whole 'to' column, which is what R's as.character()
    makes of a two-column character matrix."""
    return _str_vector([a[0] for a in arcs] + [a[1] for a in arcs])


cdef class Search:
    """The state a score-based search carries between iterations."""

    cdef SEXP data
    cdef SEXP nodes
    cdef SEXP cache
    cdef SEXP reference
    cdef SEXP blmat
    cdef SEXP wlmat
    cdef SEXP score
    cdef SEXP extra
    cdef SEXP tabu_list
    cdef SEXP amat_buf
    cdef int n
    cdef int tabu_size
    cdef bint live
    cdef object node_names

    def __cinit__(self):
        self.live = False

    def __init__(self, data, node_names, score, extra_args,
                 blacklist_amat, whitelist_amat):
        cdef SEXP dim
        cdef int i

        _ensure_init()
        pybn_arena_push()
        self.live = True

        self.node_names = [str(v) for v in node_names]
        self.n = len(self.node_names)
        self.data = _dataframe(data)
        self.nodes = _str_vector(self.node_names)
        self.score = _py_to_sexp(score)
        self.extra = _py_to_sexp(extra_args or {})
        self.blmat = _int_matrix(blacklist_amat)
        self.wlmat = _int_matrix(whitelist_amat)

        self.cache = Rf_allocVector(REALSXP, self.n * self.n)
        for i in range(self.n * self.n):
            REAL(self.cache)[i] = 0.0
        dim = Rf_allocVector(INTSXP, 2)
        INTEGER(dim)[0] = self.n
        INTEGER(dim)[1] = self.n
        Rf_setAttrib(self.cache, R_DimSymbol, dim)

        self.reference = _real_named(np.zeros(self.n), self.node_names)

        self.tabu_list = NULL
        self.amat_buf = NULL
        self.tabu_size = 0

    def enable_tabu(self, int size):
        """Allocate the tabu list.

        tabu_hash() writes each visited network's hash into a slot of this
        list, and tabu_step() reads the whole list back to know which moves
        would revisit one -- so, like the cache, it has to be the same object
        across the whole search.
        """
        cdef int i
        if size < 1:
            raise ValueError("the tabu list must have at least one slot")
        cdef SEXP dim
        self.tabu_size = size
        self.tabu_list = Rf_allocVector(VECSXP, size)
        for i in range(size):
            Rf_SET_VECTOR_ELT(self.tabu_list, i, R_NilValue)

        # A session-owned adjacency matrix for hash_network() to fill.  The
        # hash tabu_hash() computes is stored *into* the tabu list, so it must
        # outlive the call -- which rules out building the matrix in a nested
        # arena, because popping that arena would free the hash along with it
        # and leave the list holding dangling pointers.  Reusing one buffer
        # keeps the hash in the session arena without leaking a matrix per
        # iteration.
        self.amat_buf = Rf_allocVector(INTSXP, self.n * self.n)
        dim = Rf_allocVector(INTSXP, 2)
        INTEGER(dim)[0] = self.n
        INTEGER(dim)[1] = self.n
        Rf_setAttrib(self.amat_buf, R_DimSymbol, dim)

    def hash_network(self, amat, int current):
        """tabu_hash(): record the current network in slot `current`.

        Deliberately not wrapped in a nested arena: the hash this computes is
        stored in the tabu list and has to survive until the search ends.
        """
        cdef SEXP a[4]
        cdef cnp.ndarray[cnp.int32_t, ndim=2] m
        cdef int i, j
        if self.tabu_list == NULL:
            raise RuntimeError("enable_tabu() has not been called")

        m = np.ascontiguousarray(amat, dtype=np.int32)
        for j in range(self.n):
            for i in range(self.n):
                INTEGER(self.amat_buf)[j * self.n + i] = m[i, j]

        a[0] = self.amat_buf
        a[1] = self.nodes
        a[2] = self.tabu_list
        a[3] = Rf_ScalarInteger(current)
        _guarded(<void *>tabu_hash, a, 4)

    def tabu_best_step(self, amat, added, nparents, double maxp,
                       int current, double baseline):
        """tabu_step(): the best move that does not return to a network in
        the tabu list.

        With `baseline` at 0 this only accepts moves that improve the score;
        at -inf it accepts the least bad move, which is how the search escapes
        a local optimum.  Like hc_opt_step() it folds the chosen delta into the
        reference scores in place.
        """
        cdef SEXP a[13]
        if self.tabu_list == NULL:
            raise RuntimeError("enable_tabu() has not been called")
        pybn_arena_push()
        try:
            a[0] = _int_matrix(amat)
            a[1] = self.nodes
            a[2] = _int_matrix(added)
            a[3] = self.cache
            a[4] = self.reference
            a[5] = self.wlmat
            a[6] = self.blmat
            a[7] = self.tabu_list
            a[8] = Rf_ScalarInteger(current)
            a[9] = Rf_ScalarReal(baseline)
            a[10] = _real_named(nparents, None)
            a[11] = Rf_ScalarReal(maxp)
            a[12] = Rf_ScalarLogical(0)
            result = _sexp_to_py(_guarded(<void *>tabu_step, a, 13))
            return None if result["op"] is False else result
        finally:
            pybn_arena_pop()

    def close(self):
        if self.live:
            pybn_arena_pop()
            self.live = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- the mutable state --------------------------------------------------

    def get_reference(self):
        return _read_real(self.reference, self.n)

    def set_reference(self, values):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] v = np.ascontiguousarray(
            values, dtype=np.float64)
        cdef int i
        for i in range(self.n):
            REAL(self.reference)[i] = v[i]

    def get_cache(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty(
            (self.n, self.n), dtype=np.float64)
        cdef int i, j
        for j in range(self.n):
            for i in range(self.n):
                out[i, j] = REAL(self.cache)[j * self.n + i]
        return out

    # -- graph helpers ------------------------------------------------------

    def arcs_to_amat(self, arcs):
        cdef SEXP a[2]
        pybn_arena_push()
        try:
            a[0] = _arcs_sexp(arcs)
            a[1] = self.nodes
            return _read_int_matrix(
                _guarded(<void *>arcs2amat, a, 2), self.n, self.n)
        finally:
            pybn_arena_pop()

    def acyclic(self, arcs, bint directed=True):
        cdef SEXP a[5]
        pybn_arena_push()
        try:
            a[0] = _arcs_sexp(arcs)
            a[1] = self.nodes
            a[2] = Rf_ScalarLogical(0)
            a[3] = Rf_ScalarLogical(1 if directed else 0)
            a[4] = Rf_ScalarLogical(0)
            return bool(LOGICAL(_guarded(<void *>is_acyclic, a, 5))[0])
        finally:
            pybn_arena_pop()

    # -- scoring ------------------------------------------------------------

    cdef SEXP _network(self, object arcs):
        """The bn object the score functions read: its 'nodes' element maps
        each node to its parents, which is all the scores look at."""
        cdef SEXP a2[2]
        cdef SEXP a3[3]
        cdef SEXP amat, info, net

        a2[0] = _arcs_sexp(arcs)
        a2[1] = self.nodes
        amat = _guarded(<void *>arcs2amat, a2, 2)

        a3[0] = self.nodes
        a3[1] = amat
        a3[2] = Rf_ScalarLogical(0)
        info = _guarded(<void *>cache_structure, a3, 3)

        net = Rf_allocVector(VECSXP, 2)
        Rf_SET_VECTOR_ELT(net, 0, info)
        Rf_SET_VECTOR_ELT(net, 1, _arcs_sexp(arcs))
        Rf_setAttrib(net, R_NamesSymbol, _str_vector(["nodes", "arcs"]))
        return net

    def node_scores(self, arcs, targets):
        """per.node.score(): each target node's contribution to the score."""
        cdef SEXP a[6]
        targets = [str(t) for t in targets]
        pybn_arena_push()
        try:
            a[0] = self._network(arcs)
            a[1] = self.data
            a[2] = self.score
            a[3] = _str_vector(targets)
            a[4] = self.extra
            a[5] = Rf_ScalarLogical(0)
            return _read_real(
                _guarded(<void *>per_node_score, a, 6), len(targets))
        finally:
            pybn_arena_pop()

    def fill_cache(self, arcs, updated, amat, bint equivalence,
                   bint decomposability):
        """score_cache_fill(): refresh the cached score deltas in place."""
        cdef SEXP a[13]
        pybn_arena_push()
        try:
            a[0] = self.nodes
            a[1] = self.data
            a[2] = self._network(arcs)
            a[3] = self.score
            a[4] = self.extra
            a[5] = self.reference
            a[6] = Rf_ScalarLogical(1 if equivalence else 0)
            a[7] = Rf_ScalarLogical(1 if decomposability else 0)
            a[8] = _int_vector(updated)
            a[9] = _int_matrix(amat)
            a[10] = self.cache
            a[11] = self.blmat
            a[12] = Rf_ScalarLogical(0)
            _guarded(<void *>score_cache_fill, a, 13)
        finally:
            pybn_arena_pop()

    def to_be_added(self, amat, nparents, double maxp):
        """hc_to_be_added(): which arcs are candidates for addition."""
        cdef SEXP a[7]
        pybn_arena_push()
        try:
            a[0] = _int_matrix(amat)
            a[1] = self.blmat
            a[2] = R_NilValue
            # nparents comes from colSums() in R, so it is a double vector and
            # the C side reads it with REAL().
            a[3] = _real_named(nparents, None)
            a[4] = Rf_ScalarReal(maxp)
            a[5] = self.nodes
            a[6] = Rf_ScalarLogical(0)
            return _read_int_matrix(
                _guarded(<void *>hc_to_be_added, a, 7), self.n, self.n)
        finally:
            pybn_arena_pop()

    def best_step(self, amat, added, nparents, double maxp):
        """hc_opt_step(): the best single arc operation, or None if none of
        them improves the score.

        This also folds the chosen operation's delta into the reference scores
        in place, so callers must re-read get_reference() rather than keeping
        their own copy.
        """
        cdef SEXP a[10]
        pybn_arena_push()
        try:
            a[0] = _int_matrix(amat)
            a[1] = self.nodes
            a[2] = _int_matrix(added)
            a[3] = self.cache
            a[4] = self.reference
            a[5] = self.wlmat
            a[6] = self.blmat
            a[7] = _real_named(nparents, None)
            a[8] = Rf_ScalarReal(maxp)
            a[9] = Rf_ScalarLogical(0)
            result = _sexp_to_py(_guarded(<void *>hc_opt_step, a, 10))
            # bnlearn signals "nothing improved the score" with op = FALSE.
            return None if result["op"] is False else result
        finally:
            pybn_arena_pop()


# ---------------------------------------------------------------------------
# graph utilities that do not need a search in progress
# ---------------------------------------------------------------------------

cdef extern SEXP root_nodes(SEXP bn, SEXP check)
cdef extern SEXP topological_ordering(SEXP bn, SEXP root_nodes_, SEXP reverse,
    SEXP debug)


cdef SEXP _bn_object(object node_names, object arcs) except? NULL:
    """The minimal bn object the graph routines read."""
    cdef SEXP a2[2]
    cdef SEXP a3[3]
    cdef SEXP nodes = _str_vector(node_names)
    cdef SEXP amat, info, net

    a2[0] = _arcs_sexp(arcs)
    a2[1] = nodes
    amat = _guarded(<void *>arcs2amat, a2, 2)

    a3[0] = nodes
    a3[1] = amat
    a3[2] = Rf_ScalarLogical(0)
    info = _guarded(<void *>cache_structure, a3, 3)

    net = Rf_allocVector(VECSXP, 2)
    Rf_SET_VECTOR_ELT(net, 0, info)
    Rf_SET_VECTOR_ELT(net, 1, _arcs_sexp(arcs))
    Rf_setAttrib(net, R_NamesSymbol, _str_vector(["nodes", "arcs"]))
    return net


def topological_order(node_names, arcs):
    """The node order bnlearn's topological.ordering() produces.

    R sorts the nodes by the depth the C routine assigns them, and R's sort()
    is stable, so nodes at the same depth keep the order they have in the data.
    modelstring() depends on this, so it is reproduced rather than approximated.
    """
    cdef SEXP a2[2]
    cdef SEXP a4[4]
    cdef SEXP bn, roots, depths
    cdef int i

    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        bn = _bn_object(node_names, arcs)

        a2[0] = bn
        a2[1] = Rf_ScalarLogical(0)
        roots = _guarded(<void *>root_nodes, a2, 2)

        a4[0] = bn
        a4[1] = roots
        a4[2] = Rf_ScalarLogical(0)
        a4[3] = Rf_ScalarLogical(0)
        depths = _guarded(<void *>topological_ordering, a4, 4)

        # read through _sexp_to_py rather than assuming a type: this returns
        # an *integer* vector, and reading it as doubles yields garbage that
        # sorts into a plausible-looking but wrong order.
        depth_by_node = _sexp_to_py(depths)
    finally:
        pybn_arena_pop()

    names = list(depth_by_node)
    values = np.asarray([depth_by_node[k] for k in names])

    # R's sort() is stable, so nodes at equal depth keep their original order.
    return [names[i] for i in np.argsort(values, kind="stable")]


# ---------------------------------------------------------------------------
# constraint-based structure learning
#
# These algorithms are driven from Python -- the grow/shrink loops are plain
# control flow -- but every conditional independence test goes through the C
# core, and there are a great many of them.  The data frame is therefore
# converted once, into the arena this object owns, rather than per test.
# ---------------------------------------------------------------------------

cdef extern SEXP allsubs_test(SEXP x, SEXP y, SEXP sx, SEXP fixed, SEXP data,
    SEXP test, SEXP alpha, SEXP extra_args, SEXP min, SEXP max, SEXP complete,
    SEXP debug)
cdef extern SEXP bn_recovery(SEXP bn, SEXP mb, SEXP filter, SEXP debug)
cdef extern SEXP roundrobin_test(SEXP x, SEXP z, SEXP fixed, SEXP data,
    SEXP test, SEXP alpha, SEXP extra_args, SEXP complete, SEXP debug)
cdef extern SEXP cpdag(SEXP arcs, SEXP nodes, SEXP moral, SEXP fix, SEXP wlbl,
    SEXP whitelist, SEXP blacklist, SEXP illegal, SEXP debug)
cdef extern SEXP amat2arcs(SEXP amat, SEXP nodes)
cdef extern SEXP vstructures(SEXP bn, SEXP arcs, SEXP moral, SEXP debug)


cdef object _arcs_from_sexp(SEXP x):
    """A two-column character matrix comes back as 'from' then 'to'."""
    cdef int n = Rf_length(x) // 2
    cdef int i
    return [(CHAR(Rf_STRING_ELT(x, i)).decode("utf-8"),
             CHAR(Rf_STRING_ELT(x, i + n)).decode("utf-8"))
            for i in range(n)]


cdef class Tester:
    """Repeated conditional independence tests over one data set."""

    cdef SEXP data
    cdef SEXP complete
    cdef SEXP test
    cdef SEXP extra
    cdef SEXP nodes
    cdef SEXP dsep_symbol
    cdef bint live
    cdef object node_names

    def __cinit__(self):
        self.live = False

    def __init__(self, data, test, extra_args=None):
        _ensure_init()
        pybn_arena_push()
        self.live = True

        self.node_names = [str(c) for c in data.columns]
        self.data = _dataframe(data)
        self.nodes = _str_vector(self.node_names)
        self.test = _py_to_sexp(str(test))
        self.extra = _py_to_sexp(extra_args or {})
        self.complete = _named_logical(self.node_names, True)
        self.dsep_symbol = Rf_install(b"dsep.set")

    def close(self):
        if self.live:
            pybn_arena_pop()
            self.live = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def pvalue(self, x, y, sx=None):
        """indep.test() in learning mode: just the p-value."""
        cdef SEXP a[9]
        cdef SEXP out
        pybn_arena_push()
        try:
            a[0] = _str_vector([str(x)])
            a[1] = _str_vector([str(y)])
            a[2] = _str_vector([str(v) for v in sx]) if sx else R_NilValue
            a[3] = self.data
            a[4] = self.test
            a[5] = Rf_ScalarReal(1.0)
            a[6] = self.extra
            a[7] = Rf_ScalarLogical(1)          # learning = TRUE
            a[8] = self.complete
            out = _guarded(<void *>indep_test, a, 9)
            return REAL(out)[0]
        finally:
            pybn_arena_pop()

    def pvalues(self, xs, y, sx=None):
        """indep.test() with a vector of candidates: the p-value of each x
        against y given sx.  The IAMB family picks the strongest association
        from this in every forward step, so it is worth doing in one call."""
        cdef SEXP a[9]
        cdef SEXP out
        cdef int i
        xs = [str(v) for v in xs]
        if not xs:
            return {}
        pybn_arena_push()
        try:
            a[0] = _str_vector(xs)
            a[1] = _str_vector([str(y)])
            a[2] = _str_vector([str(v) for v in sx]) if sx else R_NilValue
            a[3] = self.data
            a[4] = self.test
            a[5] = Rf_ScalarReal(1.0)
            a[6] = self.extra
            a[7] = Rf_ScalarLogical(1)          # learning = TRUE
            a[8] = self.complete
            out = _guarded(<void *>indep_test, a, 9)
            return {xs[i]: REAL(out)[i] for i in range(len(xs))}
        finally:
            pybn_arena_pop()

    def roundrobin(self, x, z, fixed=None, double alpha=1.0):
        """roundrobin.test(): test x against each element of z given the rest,
        which is IAMB's backward phase."""
        cdef SEXP a[9]
        cdef SEXP out
        z = [str(v) for v in (z or [])]
        if not z:
            return {}
        pybn_arena_push()
        try:
            a[0] = _str_vector([str(x)])
            a[1] = _str_vector(z)
            a[2] = _str_vector([str(v) for v in (fixed or [])])
            a[3] = self.data
            a[4] = self.test
            a[5] = Rf_ScalarReal(alpha)
            a[6] = self.extra
            a[7] = self.complete
            a[8] = Rf_ScalarLogical(0)
            out = _guarded(<void *>roundrobin_test, a, 9)
            return _sexp_to_py(out)
        finally:
            pybn_arena_pop()

    def allsubs(self, x, y, sx=None, fixed=None, double alpha=1.0,
                int min=0, int max=-1):
        """allsubs.test(): test x against y over every subset of sx, stopping
        as soon as one of them separates them.

        Returns the three p-values plus the separating set, which the
        v-structure detection later needs.
        """
        cdef SEXP a[12]
        cdef SEXP out, dsep
        sx = [str(v) for v in (sx or [])]
        fixed = [str(v) for v in (fixed or [])]
        if max < 0:
            max = len(sx)
        max = min_int(max, len(sx))

        pybn_arena_push()
        try:
            a[0] = _str_vector([str(x)])
            a[1] = _str_vector([str(y)])
            a[2] = _str_vector(fixed + sx)
            a[3] = _str_vector(fixed) if fixed else _str_vector([])
            a[4] = self.data
            a[5] = self.test
            a[6] = Rf_ScalarReal(alpha)
            a[7] = self.extra
            a[8] = Rf_ScalarInteger(min)
            a[9] = Rf_ScalarInteger(max)
            a[10] = self.complete
            a[11] = Rf_ScalarLogical(0)
            out = _guarded(<void *>allsubs_test, a, 12)

            result = {
                "p.value": REAL(out)[0],
                "min.p.value": REAL(out)[1],
                "max.p.value": REAL(out)[2],
            }
            dsep = Rf_getAttrib(out, self.dsep_symbol)
            result["dsep.set"] = (
                None if dsep == R_NilValue
                else [CHAR(Rf_STRING_ELT(dsep, i)).decode("utf-8")
                      for i in range(Rf_length(dsep))])
            return result
        finally:
            pybn_arena_pop()


cdef int min_int(int a, int b):
    return a if a < b else b


def recover_structure(structure, node_names, bint markov_blankets,
                      filter="AND"):
    """bn.recovery(): make the learned sets symmetric.

    Independence tests are run separately for each node, so the results need
    not agree -- x can end up in y's Markov blanket without y being in x's.
    This is what reconciles them.
    """
    cdef SEXP a[4]
    cdef SEXP bn, out
    cdef int i

    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        bn = Rf_allocVector(VECSXP, len(node_names))
        for i, name in enumerate(node_names):
            entry = structure[name]
            if isinstance(entry, dict):
                item = Rf_allocVector(VECSXP, 2)
                Rf_SET_VECTOR_ELT(item, 0, _str_vector(entry.get("mb", [])))
                Rf_SET_VECTOR_ELT(item, 1, _str_vector(entry.get("nbr", [])))
                Rf_setAttrib(item, R_NamesSymbol, _str_vector(["mb", "nbr"]))
                Rf_SET_VECTOR_ELT(bn, i, item)
            else:
                Rf_SET_VECTOR_ELT(bn, i, _str_vector(entry))
        Rf_setAttrib(bn, R_NamesSymbol, _str_vector(node_names))

        a[0] = bn
        a[1] = Rf_ScalarLogical(1 if markov_blankets else 0)
        a[2] = Rf_ScalarInteger(1 if filter == "OR" else 2)
        a[3] = Rf_ScalarLogical(0)
        out = _guarded(<void *>bn_recovery, a, 4)

        # Read this explicitly rather than through _sexp_to_py: that unwraps a
        # length-one vector into a scalar, which would turn a one-node Markov
        # blanket into a bare string.  Iterating it then yields characters, and
        # for a variable named "M. Work" the caller sees "M".
        return _read_node_sets(out, node_names, markov_blankets)
    finally:
        pybn_arena_pop()


cdef object _strings(SEXP x):
    cdef int i
    if x == R_NilValue:
        return []
    return [CHAR(Rf_STRING_ELT(x, i)).decode("utf-8")
            for i in range(Rf_length(x))]


cdef object _read_node_sets(SEXP out, object node_names, bint flat):
    """bn.recovery() returns either a plain character vector per node (Markov
    blankets) or a two-element list of mb and nbr."""
    cdef int i
    cdef SEXP entry
    result = {}
    for i, name in enumerate(node_names):
        entry = Rf_VECTOR_ELT(out, i)
        if TYPEOF(entry) == VECSXP:
            result[name] = {"mb": _strings(Rf_VECTOR_ELT(entry, 0)),
                            "nbr": _strings(Rf_VECTOR_ELT(entry, 1))}
        else:
            result[name] = _strings(entry)
    return result


def cpdag_arcs(arcs, node_names, whitelist=None, blacklist=None,
               illegal=None, bint moral=False, bint fix=False,
               bint wlbl=False):
    """cpdag(): propagate arc directions, and optionally complete the graph
    into a DAG (`fix`)."""
    cdef SEXP a[9]
    cdef SEXP out, nodes

    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        nodes = _str_vector(node_names)
        a[0] = _arcs_sexp(arcs)
        a[1] = nodes
        a[2] = Rf_ScalarLogical(1 if moral else 0)
        a[3] = Rf_ScalarLogical(1 if fix else 0)
        a[4] = Rf_ScalarLogical(1 if wlbl else 0)
        a[5] = _arcs_sexp(whitelist) if whitelist else R_NilValue
        a[6] = _arcs_sexp(blacklist) if blacklist else R_NilValue
        a[7] = _arcs_sexp(illegal) if illegal else R_NilValue
        a[8] = Rf_ScalarLogical(0)
        out = _guarded(<void *>cpdag, a, 9)

        a[0] = out
        a[1] = nodes
        return _arcs_from_sexp(_guarded(<void *>amat2arcs, a, 2))
    finally:
        pybn_arena_pop()


# ---------------------------------------------------------------------------
# graph utilities and the pairwise (mutual-information) learners
#
# These are thin: the work is already done by entry points bnlearn exports, so
# what is left is building the bn object they read and converting the arc set
# they return.
# ---------------------------------------------------------------------------

cdef extern SEXP dag2ug(SEXP bn, SEXP moral, SEXP debug)
cdef extern SEXP shd(SEXP learned, SEXP golden, SEXP debug)
cdef extern SEXP pdag2dag(SEXP arcs, SEXP nodes)
cdef extern SEXP nparams_structure(SEXP graph, SEXP data, SEXP estimator,
    SEXP debug)
cdef extern SEXP chow_liu(SEXP data, SEXP nodes, SEXP estimator,
    SEXP whitelist, SEXP blacklist, SEXP complete, SEXP conditional,
    SEXP debug)
cdef extern SEXP aracne(SEXP data, SEXP estimator, SEXP whitelist,
    SEXP blacklist, SEXP complete, SEXP debug)


def undirected_arcs(node_names, arcs, bint moral=False):
    """dag2ug(): drop the directions, optionally moralising first."""
    cdef SEXP a[3]
    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _bn_object(node_names, arcs)
        a[1] = Rf_ScalarLogical(1 if moral else 0)
        a[2] = Rf_ScalarLogical(0)
        return _arcs_from_sexp(_guarded(<void *>dag2ug, a, 3))
    finally:
        pybn_arena_pop()


def structural_hamming(node_names, learned_arcs, true_arcs):
    """shd(): the count of arcs that differ between two graphs."""
    cdef SEXP a[3]
    cdef SEXP out
    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _bn_object(node_names, learned_arcs)
        a[1] = _bn_object(node_names, true_arcs)
        a[2] = Rf_ScalarLogical(0)
        out = _guarded(<void *>shd, a, 3)
        return int(REAL(out)[0]) if TYPEOF(out) == REALSXP else INTEGER(out)[0]
    finally:
        pybn_arena_pop()


def extend_pdag(arcs, ordering):
    """pdag2dag(): orient the remaining undirected arcs consistently with a
    node ordering."""
    cdef SEXP a[2]
    ordering = [str(v) for v in ordering]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _arcs_sexp(arcs)
        a[1] = _str_vector(ordering)
        return _arcs_from_sexp(_guarded(<void *>pdag2dag, a, 2))
    finally:
        pybn_arena_pop()


def count_parameters(node_names, arcs, data, estimator="bic"):
    """nparams(): the number of free parameters the network implies."""
    cdef SEXP a[4]
    cdef SEXP out
    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _bn_object(node_names, arcs)
        a[1] = _dataframe(data)
        a[2] = _py_to_sexp(str(estimator))
        a[3] = Rf_ScalarLogical(0)
        out = _guarded(<void *>nparams_structure, a, 4)
        return float(REAL(out)[0]) if TYPEOF(out) == REALSXP \
            else float(INTEGER(out)[0])
    finally:
        pybn_arena_pop()


def chow_liu_arcs(data, node_names, estimator, whitelist=None, blacklist=None):
    """chow.liu(): the maximum-weight spanning tree over mutual information."""
    cdef SEXP a[8]
    node_names = [str(v) for v in node_names]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _dataframe(data)
        a[1] = _str_vector(node_names)
        a[2] = _py_to_sexp(str(estimator))
        a[3] = _arcs_sexp(whitelist) if whitelist else R_NilValue
        a[4] = _arcs_sexp(blacklist) if blacklist else R_NilValue
        a[5] = _named_logical(node_names, True)
        a[6] = R_NilValue
        a[7] = Rf_ScalarLogical(0)
        return _arcs_from_sexp(_guarded(<void *>chow_liu, a, 8))
    finally:
        pybn_arena_pop()


def aracne_arcs(data, estimator, whitelist=None, blacklist=None):
    """aracne(): mutual information filtered by the data processing
    inequality."""
    cdef SEXP a[6]
    node_names = [str(c) for c in data.columns]

    _ensure_init()
    pybn_arena_push()
    try:
        a[0] = _dataframe(data)
        a[1] = _py_to_sexp(str(estimator))
        a[2] = _arcs_sexp(whitelist) if whitelist else R_NilValue
        a[3] = _arcs_sexp(blacklist) if blacklist else R_NilValue
        a[4] = _named_logical(node_names, True)
        a[5] = Rf_ScalarLogical(0)
        return _arcs_from_sexp(_guarded(<void *>aracne, a, 6))
    finally:
        pybn_arena_pop()
