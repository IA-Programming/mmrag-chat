#!/usr/bin/env bash
# Builds llama-server inside the Debian 11 toolchain container (docker/Dockerfile.build).
# Invoked by tools/llama/build_llama.py; can also run in CI. DEV ONLY (DR-001).
#
# Inputs (env): SRC (llama.cpp checkout), OUT (output dir), JOBS, STATIC_STDLIB (1|0)
# Output: $OUT/pkg/  (llama-server + shared libs + CPU variant backends, RPATH=$ORIGIN)
set -euo pipefail

SRC=${SRC:-/src/llama.cpp}
OUT=${OUT:-/out}
JOBS=${JOBS:-$(nproc)}
STATIC_STDLIB=${STATIC_STDLIB:-1}
export CC=${CC:-clang-18} CXX=${CXX:-clang++-18}

link_flags="-fuse-ld=lld"
if [ "$STATIC_STDLIB" = "1" ]; then
    # libstdc++/libgcc linked into every ELF so the package only needs glibc.
    link_flags="$link_flags -static-libstdc++ -static-libgcc"
fi

# Runtime-selected CPU variants (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS): one package
# picks the best kernel set for whatever CPU Community Cloud gives us. No -march=native.
cmake -S "$SRC" -B "$OUT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_OPENMP=OFF \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="$link_flags" \
    -DCMAKE_SHARED_LINKER_FLAGS="$link_flags" \
    -DCMAKE_MODULE_LINKER_FLAGS="$link_flags" \
    -DLLAMA_BUILD_NUMBER="${LLAMA_BUILD_NUMBER:-0}" \
    -DLLAMA_BUILD_COMMIT="${LLAMA_BUILD_COMMIT:-unknown}"

cmake --build "$OUT/build" --target llama-server -j "$JOBS"
# CPU variant backends are separate targets when GGML_BACKEND_DL is on.
mapfile -t cpu_targets < <(cmake --build "$OUT/build" --target help | grep -oE 'ggml-cpu-[a-z0-9_]+' | sort -u)
if [ "${#cpu_targets[@]}" -gt 0 ]; then
    cmake --build "$OUT/build" --target "${cpu_targets[@]}" -j "$JOBS"
fi

rm -rf "$OUT/pkg" && mkdir -p "$OUT/pkg"
cp "$OUT/build/bin/llama-server" "$OUT/pkg/"
find "$OUT/build/bin" -maxdepth 1 -name '*.so*' -exec cp -P {} "$OUT/pkg/" \;
cp "$SRC/LICENSE" "$OUT/pkg/LICENSE-llama.cpp"

# Facts for build-info.json (parsed by build_llama.py)
{
    echo "compiler=$($CXX --version | head -1)"
    echo "glibc_build=$(ldd --version | head -1)"
    echo "cmake=$(cmake --version | head -1)"
    echo "static_stdlib=$STATIC_STDLIB"
} > "$OUT/pkg/.toolchain"
ls -la "$OUT/pkg"
