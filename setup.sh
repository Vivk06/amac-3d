#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
venv="$repo/.venv"
manifest="$venv/.setup-manifest"
esp_version="${ESP_VERSION:-129}"
esp_dir="$repo/ESP${esp_version}"
esp_root="$esp_dir/EngSketchPad"
downloads="$repo/downloads"
tarball="$downloads/ESP${esp_version}-linux-x86_64.tgz"
url="https://acdl.mit.edu/ESP/PreBuilts/ESP${esp_version}-linux-x86_64.tgz"
want_esp=0 force=0 clean=0 yes=0
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage: bash setup.sh [--esp] [--force] [--clean] [--yes]
  --esp    Install ESP/pyEGADS as well as the Python environment
  --force  Download a fresh ESP tarball and re-extract it (requires --esp)
  --clean  List paths created by setup; delete only with --yes
  --yes    Confirm --clean deletion
EOF
}
for arg in "$@"; do
    case "$arg" in
        --esp) want_esp=1 ;;
        --force) force=1 ;;
        --clean) clean=1 ;;
        --yes) yes=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $arg" ;;
    esac
done
[[ ! -n "${CONDA_PREFIX:-}" ]] || die 'deactivate conda before using amac-3d.'
[[ "$esp_version" =~ ^[0-9]+$ ]] || die 'ESP_VERSION must be numeric'
(( !force || want_esp )) || die '--force requires --esp'
(( !yes || clean )) || die '--yes requires --clean'
(( !clean || (!want_esp && !force) )) || die '--clean cannot be combined with --esp or --force'

# Every manifest entry is a fixed project path. Refuse symlink parents and alien entries.
safe_entry() {
    local p="$1"
    case "$p" in
        "$venv"|"$esp_dir"|"$downloads"|"$tarball") ;;
        *) die "unsafe manifest path: $p" ;;
    esac
    [[ "$p" == "$repo/"* && ! -L "$p" ]] || die "unsafe project path: $p"
    [[ ! -L "$repo/downloads" && ! -L "$repo/.venv" ]] || die 'symlinked setup directory'
}
record() {
    safe_entry "$1"
    touch "$manifest"
    grep -Fxq -- "$1" "$manifest" || printf '%s\n' "$1" >> "$manifest"
}

if (( clean )); then
    [[ -f "$manifest" ]] || { echo 'No setup manifest found.'; exit 0; }
    mapfile -t paths < "$manifest"
    for p in "${paths[@]}"; do
        safe_entry "$p"
        [[ -e "$p" ]] || continue
        printf '%s (%s bytes)\n' "$p" "$(du -sb -- "$p" | cut -f1)"
    done
    if (( !yes )); then echo 'Dry run; pass --yes to delete these paths.'; exit 0; fi
    # Delete children first; each path has passed the fixed-name, in-project checks.
    for p in "$tarball" "$esp_dir" "$downloads" "$venv"; do
        for entry in "${paths[@]}"; do
            [[ "$entry" == "$p" && -e "$p" ]] || continue
            safe_entry "$p"
            if [[ "$p" == "$downloads" ]]; then rmdir -- "$p" 2>/dev/null || die 'downloads is not empty';
            else rm -rf -- "$p"; fi
        done
    done
    exit 0
fi

python_cmd="${PYTHON:-}"
if [[ -z "$python_cmd" ]]; then
    for candidate in python3.14 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -I -c 'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)' 2>/dev/null; then
            python_cmd="$candidate"; break
        fi
    done
fi
[[ -n "$python_cmd" ]] || die 'CPython 3.14 not found; set PYTHON to its executable'
"$python_cmd" -I -c 'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)' || die 'PYTHON must be CPython 3.14'
if [[ ! -d "$venv" ]]; then
    "$python_cmd" -m venv "$venv"
    record "$venv"
fi
py="$venv/bin/python"
[[ -x "$py" ]] || die 'venv interpreter missing'
"$py" -m pip install --no-cache-dir -r "$repo/requirements.txt"
venv_tbb="$venv/lib/libtbb.so.12"
tbb_version() {
    "$py" -I -c 'import ctypes,sys; print(ctypes.CDLL(sys.argv[1]).TBB_runtime_interface_version())' "$1" 2>/dev/null || true
}
pip_tbb="$(tbb_version "$venv_tbb")"
[[ "$pip_tbb" =~ ^[0-9]+$ ]] && (( pip_tbb >= 12060 )) || die 'venv TBB interface is below 12060'
system_tbb="$(env -u LD_LIBRARY_PATH "$python_cmd" -I -S -c 'import ctypes; print(ctypes.CDLL("libtbb.so.12").TBB_runtime_interface_version())' 2>/dev/null || true)"
echo "TBB interface: system=${system_tbb:-unavailable}, venv=$pip_tbb"
if [[ ! "$system_tbb" =~ ^[0-9]+$ ]] || (( system_tbb < 12060 )); then
    purelib="$("$py" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    shim="$purelib/sitecustomize.py"
    if [[ -e "$shim" ]] && ! grep -Fq '# amac-3d TBB shim' "$shim"; then
        die "existing sitecustomize.py is not owned by setup: $shim"
    fi
    cat > "$shim" <<'SHIM'
# amac-3d TBB shim
try:
    import ctypes, os, sys
    _lib = os.path.join(sys.prefix, 'lib', 'libtbb.so.12')
    if os.path.exists(_lib):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
SHIM
    echo 'Installed TBB startup shim.'
fi

export PATH="$venv/bin:$PATH"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export NUMBA_THREADING_LAYER=tbb
export LD_LIBRARY_PATH="$venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPYCACHEPREFIX="$venv/pycache"
export NUMBA_CACHE_DIR="$venv/numba-cache"

if (( want_esp )); then
    command -v tar >/dev/null || die 'tar is required for ESP'
    if ! command -v curl >/dev/null && ! command -v wget >/dev/null; then die 'curl or wget is required for ESP'; fi
    [[ ! -L "$downloads" && ! -L "$esp_dir" ]] || die 'symlinked ESP path'
    cas_dir=( "$esp_dir"/OpenCASCADE-* )
    if [[ ! -f "$esp_root/lib/libegads.so" || ! -d "${cas_dir[0]}" ]] || (( force )); then
        if [[ ! -d "$downloads" ]]; then mkdir "$downloads"; record "$downloads"; fi
        if (( force )) || [[ ! -f "$tarball" ]] || ! tar tzf "$tarball" >/dev/null 2>&1; then
            part="$tarball.part"
            [[ ! -L "$part" ]] || die "symlinked download path: $part"
            if command -v curl >/dev/null; then curl -fsSL -o "$part" "$url" || die "download failed; cached tarball kept at $tarball";
            else wget -q -O "$part" "$url" || die "download failed; cached tarball kept at $tarball"; fi
            tar tzf "$part" >/dev/null || die "downloaded archive invalid; cached tarball kept at $tarball"
            mv -f "$part" "$tarball"
            record "$tarball"
        fi
        if (( force )) && [[ -d "$esp_dir" ]]; then safe_entry "$esp_dir"; rm -rf -- "$esp_dir"; fi
        tar xzf "$tarball" -C "$repo" || die "extraction failed; tarball kept at $tarball"
        [[ -d "$esp_dir" ]] || die "extraction did not create $esp_dir; tarball kept"
        record "$esp_dir"
        cas_dir=( "$esp_dir"/OpenCASCADE-* )
    fi
    cas_root="${cas_dir[0]}"
    [[ -f "$esp_root/lib/libegads.so" && -f "$esp_root/lib/libocsm.so" && -d "$esp_root/pyESP/pyEGADS" && -d "$cas_root" ]] || die "ESP structure incomplete; tarball kept at $tarball"
    export ESP_ROOT="$esp_root" CASROOT="$cas_root" ESP_ARCH=LINUX64 CASARCH=.
    cas_version="${cas_root##*/OpenCASCADE-}"
    IFS=. read -r cas_major cas_minor _ <<< "$cas_version"
    export CASREV="$cas_major.$cas_minor"
    export LD_LIBRARY_PATH="$venv/lib:$cas_root/lib:$esp_root/lib"
    export PYTHONPATH="$repo/src:$repo:$esp_root/pyESP:$esp_root/lib"
    "$py" -c 'from pyEGADS import egads; ctx=egads.Context(); print("pyEGADS: OK")' || die "pyEGADS import failed; tarball kept at $tarball"
    if [[ -f "$tarball" ]]; then
        safe_entry "$tarball"
        bytes="$(stat -c %s -- "$tarball")"
        rm -- "$tarball"
        rmdir -- "$downloads" 2>/dev/null || die "downloads is not empty after archive cleanup: $downloads"
        echo "ESP archive cleanup: freed $bytes bytes; downloads/ is absent."
    fi
fi

cd "$repo"
"$py" tests/regression.py --lane fast
echo 'Setup complete. Start a new shell with: source activate.sh'
