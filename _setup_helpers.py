import importlib
import os
import shutil
import subprocess
import sys
from typing import Optional, cast

from setuptools import Command, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.sdist import sdist

try:
    _bdist_wheel_module = importlib.import_module(
        "setuptools.command.bdist_wheel"
    )
except Exception:
    _bdist_wheel_module = importlib.import_module("wheel.bdist_wheel")

bdist_wheel_base = cast(type, _bdist_wheel_module.bdist_wheel)

protocol_compiler: Optional[str] = None
cuda_home_path: Optional[str] = None
torch_ready = False


def check_nvcc_installed(cuda_home: str) -> None:
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
    except Exception as exc:
        raise RuntimeError(
            "nvcc is not installed or not found in PATH. "
            "Please ensure CUDA toolkit is installed and nvcc is available."
        ) from exc


def ensure_torch_environment() -> None:
    global protocol_compiler
    global cuda_home_path
    global torch_ready

    if torch_ready:
        return

    try:
        torch = importlib.import_module("torch")
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    except Exception as exc:
        print(
            "[WARNING] Unable to import torch, pre-compiling ops is disabled. "
            "Please visit https://pytorch.org/ to install torch."
        )
        raise exc

    torch_path = str(getattr(cpp_extension, "_TORCH_PATH"))
    exec_ext = str(getattr(cpp_extension, "EXEC_EXT"))
    cuda_home = getattr(cpp_extension, "CUDA_HOME")

    protocol_compiler = os.path.join(torch_path, "bin", "protoc" + exec_ext)
    cuda_home_path = None if cuda_home is None else str(cuda_home)

    print(f"torch version: {torch.__version__}")

    assert cuda_home_path is not None, "CUDA_HOME is not set"
    check_nvcc_installed(cuda_home_path)
    torch_ready = True


def is_ninja_available() -> bool:
    try:
        _ = subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False
    return True


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class cmake_build_ext(build_ext):
    did_config: dict[str, bool] = {}

    def run(self):
        ensure_torch_environment()
        if not self.extensions:
            self.extensions = [
                Extension("morphling._C", sources=[]),
                Extension("morphling._Msg", sources=[]),
                Extension("morphling._GreenCtx", sources=[]),
            ]
        super().run()

    def configure(self, ext: Extension) -> None:
        cmake_lists_dir = os.path.abspath(getattr(ext, "cmake_lists_dir", "."))
        if cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={self.build_temp}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DMORPHLING_PYTHON_EXECUTABLE={sys.executable}",
        ]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile=8",
            ]
        else:
            build_tool = []

        if os.environ.get("TEST") == "1":
            cmake_args.append("-DBUILD_TESTS=ON")
        else:
            cmake_args.append("-DBUILD_TESTS=OFF")

        ccache = shutil.which("ccache")
        if ccache:
            cmake_args += [
                f"-DCMAKE_C_COMPILER_LAUNCHER={ccache}",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache}",
                f"-DCMAKE_CUDA_COMPILER_LAUNCHER={ccache}",
            ]

        _ = subprocess.check_call(
            ["cmake", cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

        if os.environ.get("TEST") == "1":
            test_folder = os.path.join(self.build_temp, "tests", "cpp")
            with open(
                os.path.join(test_folder, "CTestTestfile.cmake"),
                "r",
                encoding="utf-8",
            ) as test_file:
                for line in test_file.readlines():
                    if "add_test(" in line:
                        test_name = (
                            line.strip()
                            .split("add_test([=[")[-1]
                            .split("]=]")[0]
                        )
                        _ = subprocess.check_call(
                            ["ninja", "-C", self.build_temp, test_name]
                        )

    def build_extensions(self) -> None:
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("Cannot find CMake executable") from exc

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        for ext in self.extensions:
            self.configure(ext)
            ext_target_name = remove_prefix(ext.name, "morphling.")
            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                "32",
            ]
            _ = subprocess.check_call(
                ["cmake", *build_args], cwd=self.build_temp
            )
            print(self.build_temp, ext_target_name)

        self.copy_extensions_to_source()

    def copy_extensions_to_source(self) -> None:
        """Copy built .so extensions into the source package directory.

        This is needed when the working directory is the project root (e.g.,
        in Docker with WORKDIR /app), where Python finds the source package
        before the installed one in site-packages.  Without this step,
        ``import morphling._C`` fails because the .so files only exist in
        the install tree, not next to __init__.py in the source tree.
        """
        build_lib = self.build_lib
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src = os.path.join(build_lib, filename)
            if os.path.exists(src):
                dest = (
                    filename  # relative path, e.g. morphling/_C.cpython-...so
                )
                dest_dir = os.path.dirname(dest)
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                self.copy_file(src, dest)

    def get_ext_filename(self, fullname):
        for ext in self.extensions:
            target_type = getattr(ext, "target_type", "shared")
            if ext.name == fullname and target_type == "executable":
                return fullname.replace(".", "/")
        return super().get_ext_filename(fullname)


class BuildPackageProtos(Command):
    description = "build grpc protobuf modules"
    user_options = []
    strict_mode = False

    def initialize_options(self):
        self.strict_mode = False

    def finalize_options(self):
        return None

    def _build_package_proto(self, root: str, proto_file: str) -> None:
        # Prefer grpc_tools.protoc (libprotoc >= 4.x) so generated *_pb2.py
        # is compatible with the runtime protobuf>=4.21,<7 pin. The torch
        # bundled protoc is 3.13 and produces code that the modern runtime
        # rejects with "Descriptors cannot be created directly".
        try:
            importlib.import_module("grpc_tools.protoc")
            command = [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I",
                "./",
                f"--python_out={root}",
                proto_file,
            ]
        except ImportError:
            if protocol_compiler is None:
                raise RuntimeError("Protocol compiler path is not initialized")
            command = [
                protocol_compiler,
                "-I",
                "./",
                f"--python_out={root}",
                proto_file,
            ]
        _ = subprocess.check_call(command)

    def run(self):
        ensure_torch_environment()
        self._build_package_proto(".", "morphling/proto/morphling.proto")


class CustomInstall(install):
    def run(self):
        self.run_command("build_ext")
        self.run_command("build_package_protos")
        super().run()


class CustomBuild(sdist):
    def run(self):
        self.run_command("build_package_protos")
        super().run()


class CustomBdistWheel(bdist_wheel_base):
    def run(self):
        self.run_command("build_package_protos")
        super().run()
