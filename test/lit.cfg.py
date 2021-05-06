# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'AIE'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.aie_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.aie_obj_root, 'test')
config.aie_tools_dir = os.path.join(config.aie_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
#llvm_config.with_environment('LM_LICENSE_FILE', os.getenv('LM_LICENSE_FILE'))
#llvm_config.with_environment('XILINXD_LICENSE_FILE', os.getenv('XILINXD_LICENSE_FILE'))
llvm_config.with_environment('CARDANO', config.vitis_cardano_root)
llvm_config.with_environment('VITIS_SYSROOT',config.vitis_sysroot)

#test if LM_LICENSE_FILE valid
import shutil
result = shutil.which("xchesscc")
#validLMLicense = (result != None)

import subprocess
if result != None:
    result = subprocess.run(['xchesscc','+v'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    validLMLicense = (len(result.stderr.decode('utf-8')) == 0)
else:
    validLMLicense = False

if validLMLicense:
    config.available_features.add('valid_xchess_license')
    lm_license_file = os.getenv('LM_LICENSE_FILE')
    if(lm_license_file != None):
        llvm_config.with_environment('LM_LICENSE_FILE', lm_license_file)
    xilinxd_license_file = os.getenv('XILINXD_LICENSE_FILE')
    if(xilinxd_license_file != None):
        llvm_config.with_environment('XILINXD_LICENSE_FILE', xilinxd_license_file)
else:
    print("WARNING: no valid xchess license that is required by some of the lit tests")

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir, config.vitis_aietools_root + "/bin"]
tools = [
    'aie-opt',
    'aie-translate',
    'aiecc.py',
    'ld.lld',
    'llc',
    'llvm-objdump',
    'opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
