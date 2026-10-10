#!/usr/bin/env bash
# Build the llama-server Crucible pins for cuda-linux (crucible/hosttools.py,
# LLAMA_SERVER_BUILDS) and pack it for the `tools` release.
#
# ggml-org publishes CUDA builds of llama.cpp for Windows only, so this one is ours. Only the
# binary ships: it links cudart and cuBLAS dynamically and, at run time, loads them from the
# llm env's PyPI nvidia wheels (crucible/llamacpp.py, CUDA_LINUX_LIBRARIES). The toolkit it
# is compiled with is the same family of wheels, at the versions the llm env pins
# (crucible/envs/llm/cuda-linux.txt: nvidia-cuda-runtime 13.0.96, nvidia-cublas 13.1.1.3),
# with nvcc, crt, nvvm and cccl at the matching 13.0 releases: CCCL refuses to compile when
# nvcc and the runtime headers disagree on the minor version, and the env's own nvcc is
# 13.4 against a 13.0 runtime.
#
# Usage (on an x86_64 Linux or WSL2 host with gcc, g++, cmake, git and a python3 with venv; no
# CUDA toolkit or GPU needed, nothing is installed outside WORK):
#   scripts/build-llama-server-linux.sh WORK_DIR
# Prints the archive's path, sha256 and size: the three facts the pin records.
set -euo pipefail

TAG=b10970
CUDA=13.0
ARCHS="75-real;80-real;86-real;89-real;90-real;120-real"
NAME="llama-server-${TAG}-cuda${CUDA}-linux-x86_64"
WHEELS=(
  "nvidia-cuda-nvcc==13.0.88"
  "nvidia-cuda-crt==13.0.88"
  "nvidia-nvvm==13.0.88"
  "nvidia-cuda-nvrtc==13.0.88"
  "nvidia-cuda-cccl==13.0.85"
  "nvidia-cuda-runtime==13.0.96"
  "nvidia-cublas==13.1.1.3"
)

WORK=${1:?usage: $0 WORK_DIR}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
SRC=$WORK/src
VENV=$WORK/toolkit-venv
TK=$WORK/toolkit
BUILD=$WORK/build
OUT=$WORK/out

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
COMMIT=$(git -C "$SRC" rev-parse HEAD)

"${PYTHON:-python3}" -m venv "$VENV"
"$VENV/bin/pip" install --quiet "${WHEELS[@]}"
CU=$("$VENV/bin/python" -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__path__[0]) / "cu13")')

# A toolkit-shaped directory over the wheels. bin is a real directory so nvcc.profile's
# $(_HERE_)/.. is this toolkit and not the wheel; lib carries the unversioned names CMake
# links against; the driver library (libcuda.so.1) is the host's and is never shipped.
rm -rf "$TK"
mkdir -p "$TK/bin" "$TK/lib/stubs"
for f in "$CU"/bin/*; do ln -s "$f" "$TK/bin/$(basename "$f")"; done
for d in include nvvm; do ln -s "$CU/$d" "$TK/$d"; done
for f in "$CU"/lib/*; do ln -s "$f" "$TK/lib/$(basename "$f")"; done
ln -s "$CU/lib/libcudart.so.13" "$TK/lib/libcudart.so"
ln -s "$CU/lib/libcublas.so.13" "$TK/lib/libcublas.so"
ln -s "$CU/lib/libcublasLt.so.13" "$TK/lib/libcublasLt.so"
DRIVER=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
if [ -z "$DRIVER" ] && [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then DRIVER=/usr/lib/wsl/lib/libcuda.so.1; fi
if [ -z "$DRIVER" ]; then
  echo "no libcuda.so.1 on this host to link against (the NVIDIA driver's library)" >&2
  exit 1
fi
ln -s "$DRIVER" "$TK/lib/stubs/libcuda.so"
ln -s "$TK/lib" "$TK/lib64"
export PATH="$TK/bin:$PATH" CUDA_PATH="$TK" CUDAToolkit_ROOT="$TK"

# Static llama/ggml, no OpenMP (no libgomp to find), libstdc++ and libgcc folded in, a CPU
# baseline of AVX2/FMA/F16C rather than this machine's, no OpenSSL, no web UI, no rpath:
# the libraries are found through LD_LIBRARY_PATH, which the engine sets.
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_OPENMP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DCMAKE_CUDA_COMPILER="$TK/bin/nvcc" \
  -DCUDAToolkit_ROOT="$TK" \
  -DCMAKE_LIBRARY_PATH="$TK/lib/stubs" \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
  -DCMAKE_SKIP_RPATH=ON
cmake --build "$BUILD" --target llama-server -j "$(( $(nproc) / 2 ))"

rm -rf "$OUT"
mkdir -p "$OUT/$NAME/bin"
cp "$BUILD/bin/llama-server" "$OUT/$NAME/bin/llama-server"
strip "$OUT/$NAME/bin/llama-server"
cp "$SRC/LICENSE" "$OUT/$NAME/LICENSE"
GLIBC=$(objdump -T "$OUT/$NAME/bin/llama-server" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
cat > "$OUT/$NAME/BUILD.txt" <<EOF
llama.cpp ${TAG} (${COMMIT}), llama-server only.
CUDA ${CUDA} from PyPI nvidia wheels: ${WHEELS[*]}
CUDA architectures: ${ARCHS}
Built by scripts/build-llama-server-linux.sh (github.com/telltaleatheist/crucible).
Needs at run time: libcudart.so.13, libcublas.so.13, libcublasLt.so.13 (Crucible's llm env),
libcuda.so.1 (the NVIDIA driver, 580 or newer), ${GLIBC} or newer.
EOF
tar -C "$OUT" -cJf "$OUT/$NAME.tar.xz" "$NAME"
echo "commit:  $COMMIT"
echo "archive: $OUT/$NAME.tar.xz"
echo "sha256:  $(sha256sum "$OUT/$NAME.tar.xz" | cut -d' ' -f1)"
echo "bytes:   $(stat -c %s "$OUT/$NAME.tar.xz")"
echo "needs:"
ldd "$OUT/$NAME/bin/llama-server" | sed 's/^/  /'
