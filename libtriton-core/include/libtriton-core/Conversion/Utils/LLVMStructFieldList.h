#ifndef LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTFIELDLIST_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTFIELDLIST_H_

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace libtriton::conversion::utils {

template <typename... FieldTypes> struct LLVMStructFieldList {
  static constexpr std::size_t kSize = sizeof...(FieldTypes);

  template <std::size_t Index>
  using FieldType =
      typename std::tuple_element<Index, std::tuple<FieldTypes...>>::type;
};

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTFIELDLIST_H_
