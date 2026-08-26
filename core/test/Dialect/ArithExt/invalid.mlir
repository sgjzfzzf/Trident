//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{'arithext.and_then' op expects each region to contain exactly one block}}
%result = "arithext.and_then"() ({}) : () -> i1
