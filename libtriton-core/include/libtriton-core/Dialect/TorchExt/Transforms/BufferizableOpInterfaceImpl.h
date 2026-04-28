#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace libtriton::torch_ext {

/// Register the `BufferizableOpInterface` external models for TorchExt ops.
/// Call this before running one-shot-bufferize to enable bufferization of
/// `torch_ext.torch_kernel_launch` ops that carry tensor operands.
void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace libtriton::torch_ext

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
