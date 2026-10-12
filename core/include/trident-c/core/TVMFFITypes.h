//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_C_TVMFFITYPES_H
#define TRIDENT_CORE_C_TVMFFITYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// AnyType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAnyTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAnyTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAnyTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAnyType(MlirType type);

// ArrayType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIArrayTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIArrayTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIArrayTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIArrayType(MlirType type);

// BoolType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIBoolTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIBoolTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIBoolTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIBoolType(MlirType type);

// DTypeType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIDTypeTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIDTypeTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIDTypeTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIDTypeType(MlirType type);

// DeviceType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIDeviceTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIDeviceTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIDeviceTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIDeviceType(MlirType type);

// ExceptionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIExceptionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIExceptionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIExceptionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIExceptionType(MlirType type);

// FloatType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIFloatTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIFloatTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIFloatTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIFloatType(MlirType type);

// FunctionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIFunctionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIFunctionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIFunctionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIFunctionType(MlirType type);

// IntType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIIntTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIIntTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIIntTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIIntType(MlirType type);

// NoneType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFINoneTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFINoneTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFINoneTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFINoneType(MlirType type);

// ObjectType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIObjectTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIObjectTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIObjectTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIObjectType(MlirType type);

// RawStrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIRawStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIRawStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIRawStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIRawStrType(MlirType type);

// SmallStrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFISmallStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFISmallStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFISmallStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFISmallStrType(MlirType type);

// StrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIStrType(MlirType type);

// TensorType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFITensorTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFITensorTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFITensorTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFITensorType(MlirType type);

// UnionType

/// Construct a verified union, emitting diagnostics and returning a null type
/// if the members are invalid or belong to another context.
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIUnionTypeGetChecked(
    MlirLocation location, intptr_t numTypes, const MlirType *types);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIUnionTypeGetName(void);
MLIR_CAPI_EXPORTED intptr_t mlirTVMFFIUnionTypeGetNumTypes(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIUnionTypeGetType(MlirType type,
                                                       intptr_t index);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIUnionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIUnionType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_C_TVMFFITYPES_H
