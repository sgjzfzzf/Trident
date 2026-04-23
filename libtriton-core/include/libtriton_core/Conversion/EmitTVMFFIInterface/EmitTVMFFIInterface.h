#ifndef LIBTRITON_CORE_CONVERSION_EMITTVMFFIINTERFACE_EMITTVMFFIINTERFACE_H_
#define LIBTRITON_CORE_CONVERSION_EMITTVMFFIINTERFACE_EMITTVMFFIINTERFACE_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DECL_EMITTVMFFIINTERFACE
#include "libtriton_core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_EMITTVMFFIINTERFACE
#include "libtriton_core/Conversion/Passes.h.inc"

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_CONVERSION_EMITTVMFFIINTERFACE_EMITTVMFFIINTERFACE_H_
