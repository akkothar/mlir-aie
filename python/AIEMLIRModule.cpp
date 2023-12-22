//===- AIEMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/Dialects.h"
#include "aie-c/Registration.h"
#include "aie-c/Translation.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_aie, m) {

  aieRegisterAllPasses();

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) {
        MlirDialectHandle aieHandle = mlirGetDialectHandle__aie__();
        MlirDialectHandle aiexHandle = mlirGetDialectHandle__aiex__();
        MlirDialectHandle aievecHandle = mlirGetDialectHandle__aievec__();
        mlirDialectHandleInsertDialect(aieHandle, registry);
        mlirDialectHandleInsertDialect(aiexHandle, registry);
        mlirDialectHandleInsertDialect(aievecHandle, registry);
      },
      py::arg("registry"));

  // AIE types bindings
  mlir_type_subclass(m, "ObjectFifoType", aieTypeIsObjectFifoType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoTypeGet(type));
          },
          "Get an instance of ObjectFifoType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  mlir_type_subclass(m, "ObjectFifoSubviewType", aieTypeIsObjectFifoSubviewType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoSubviewTypeGet(type));
          },
          "Get an instance of ObjectFifoSubviewType with given element type.",
          py::arg("self"), py::arg("type") = py::none());

  auto stealCStr = [](MlirStringRef mlirString) {
    if (!mlirString.data || mlirString.length == 0)
      throw std::runtime_error("couldn't translate");
    std::string cpp(mlirString.data, mlirString.length);
    free((void *)mlirString.data);
    py::handle pyS = PyUnicode_DecodeLatin1(cpp.data(), cpp.length(), nullptr);
    if (!pyS)
      throw py::error_already_set();
    return py::reinterpret_steal<py::str>(pyS);
  };

  m.def(
      "translate_aie_vec_to_cpp",
      [&stealCStr](MlirOperation op, bool aieml) {
        return stealCStr(aieTranslateAIEVecToCpp(op, aieml));
      },
      "module"_a, "aieml"_a = false);

  m.def(
      "translate_mlir_to_llvmir",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateModuleToLLVMIR(op));
      },
      "module"_a);

  m.def(
      "generate_cdo",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateToCDO(op));
      },
      "module"_a);

  m.def(
      "ipu_instgen",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateToIPU(op));
      },
      "module"_a);

  m.def(
      "generate_xaie",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateToXAIEV2(op));
      },
      "module"_a);

  m.def(
      "generate_bcf",
      [&stealCStr](MlirOperation op, int col, int row) {
        return stealCStr(aieTranslateToBCF(op, col, row));
      },
      "module"_a, "col"_a, "row"_a);
}
