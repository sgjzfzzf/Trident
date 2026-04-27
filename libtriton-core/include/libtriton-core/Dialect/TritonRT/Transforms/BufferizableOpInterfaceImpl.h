#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace libtriton::triton_rt {

/// Register the `BufferizableOpInterface` external models for TritonRT ops.
/// Call this before running one-shot-bufferize to enable bufferization of
/// `triton_rt.triton_kernel_launch` ops that carry tensor operands.
void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace libtriton::triton_rt

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
