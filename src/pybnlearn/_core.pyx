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
    cdef int n
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
