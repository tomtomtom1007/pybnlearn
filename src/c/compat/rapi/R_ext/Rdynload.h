/* pybnlearn: forwarding header, plus the dynamic-registration declarations
 * that bnlearn's globals.c refers to.  pybnlearn reaches the entry points
 * through Cython rather than through R's .Call() dispatch table, so these
 * exist only to keep that translation unit compiling. */
#include "../../rcompat.h"

typedef void *DL_FUNC;

typedef struct { const char *name; DL_FUNC fun; int numArgs; } R_CallMethodDef;
typedef struct { const char *name; DL_FUNC fun; int numArgs; } R_CMethodDef;
typedef struct { const char *name; DL_FUNC fun; int numArgs; } R_ExternalMethodDef;
typedef void DllInfo;

int  R_registerRoutines(DllInfo *info, const R_CMethodDef *croutines,
       const R_CallMethodDef *callRoutines, const void *fortranRoutines,
       const R_ExternalMethodDef *externalRoutines);
void R_useDynamicSymbols(DllInfo *info, int onoff);
void R_forceSymbols(DllInfo *info, int onoff);
