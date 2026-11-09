#!/usr/bin/env bash
# Activate the AMAC environment.
#
# Source this before any AMAC session — adaptation runs, tests, or development:
#   source activate.sh
#
# What it does:
#   1. Activates the AMAC venv (.venv/: Python 3.14 + numpy/numba/scipy/matplotlib)
#   2. If ESP/EGADS is installed, loads it (pyEGADS, OpenCASCADE libs)
#   3. Keeps ESP's bundled Python 3.12 off PATH and PYTHONPATH
#   4. Selects Numba's TBB threading layer (pip `tbb` wheel in .venv/lib)
#
# The script is repo-agnostic — it discovers all paths relative to its own location.
# Checks run first and nothing is changed unless they pass.
#
# Prerequisites:
#   - Run `bash setup.sh` (or `bash setup.sh --esp` for EGADS) first.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: activate.sh must be SOURCED, not executed."
    echo "       Use:  source ${BASH_SOURCE[0]}"
    exit 1
fi

# ── Discover repo root from script location ─────────────────────────────
AMAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_venv="$AMAC_ROOT/.venv"

# ── Helpers ──────────────────────────────────────────────────────────────

# Prepend a directory to a PATH-like variable, removing any prior occurrence.
_amac_prepend() {
    local var="$1" dir="$2" out="" entry
    [ -d "$dir" ] || return 0
    IFS=: read -ra _entries <<< "${!var:-}"
    for entry in "${_entries[@]}"; do
        [ -z "$entry" ] && continue
        [ "$entry" = "$dir" ] && continue
        out="${out:+$out:}$entry"
    done
    export "$var=$dir${out:+:$out}"
}

# Remove every PATH entry containing a substring.
_amac_strip() {
    local var="$1" pat="$2" out="" entry
    IFS=: read -ra _entries <<< "${!var:-}"
    for entry in "${_entries[@]}"; do
        [ -z "$entry" ] && continue
        case "$entry" in *"$pat"*) continue ;; esac
        out="${out:+$out:}$entry"
    done
    export "$var=$out"
}

_amac_warn() { echo "  WARNING: $*" >&2; }

# ── Checks (nothing is changed yet) ──────────────────────────────────────
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo 'ERROR: deactivate conda before using amac-3d.' >&2
    return 1 2>/dev/null || exit 1
fi

# -I ignores PYTHONPATH/PYTHONHOME, so this tests the venv itself.
if ! "$_venv/bin/python" -I -c '' &>/dev/null; then
    echo "ERROR: no working venv interpreter at $_venv/bin/python"
    echo "       Create it:  python3 -m venv $_venv && $_venv/bin/pip install -r $AMAC_ROOT/requirements.txt"
    echo "       The environment has NOT been activated."
    unset -f _amac_prepend _amac_strip _amac_warn
    unset _venv
    return 1 2>/dev/null || exit 1
fi

# Numba uses TBB for parallel kernels and needs interface >= 12060.
# Test the pip `tbb` copy in the venv by absolute path.
_tbb_ver="$("$_venv/bin/python" -I -c '
import ctypes, sys
try:
    print(ctypes.CDLL(sys.argv[1]).TBB_runtime_interface_version())
except Exception:
    pass
' "$_venv/lib/libtbb.so.12" 2>/dev/null)"
if ! [ "${_tbb_ver:-0}" -ge 12060 ] 2>/dev/null; then
    _tbb_pin="$(sed -n 's/^\(tbb==[^[:space:]#]*\).*/\1/p' "$AMAC_ROOT/requirements.txt" 2>/dev/null | head -n 1)"
    echo "ERROR: Numba's TBB threading layer needs TBB interface >= 12060 in $_venv/lib (found: ${_tbb_ver:-none})."
    echo "       Install it:  $_venv/bin/pip install ${_tbb_pin:-tbb>=2021.6}"
    echo "       The environment has NOT been activated."
    unset -f _amac_prepend _amac_strip _amac_warn
    unset _venv _tbb_ver _tbb_pin
    return 1 2>/dev/null || exit 1
fi

# ESP is installed iff libegads.so exists and an OpenCASCADE directory resolves.
_esp_dir="$AMAC_ROOT/ESP${ESP_VERSION:-129}"
_esp_root="$_esp_dir/EngSketchPad"
_cas_dirs=( "$_esp_dir"/OpenCASCADE-* )
_cas_root="${_cas_dirs[0]}"
unset _cas_dirs
_esp=0
if [ -f "$_esp_root/lib/libegads.so" ] && [ -d "$_cas_root" ]; then
    _esp=1
fi

# ── Apply (checks passed; the shell changes from here on) ────────────────
# Deactivate any venv the terminal session may have restored.
[ -n "${VIRTUAL_ENV:-}" ] && type deactivate &>/dev/null && deactivate
source "$_venv/bin/activate"

# PYTHONPATH contains src for `amac_3d` and the project root for `python -m run.*`;
# with ESP, it also contains pyESP. Inherited entries are discarded.
_pp_dropped=0
[ -n "${PYTHONPATH:-}" ] && _pp_dropped=1
export PYTHONPATH="$AMAC_ROOT/src:$AMAC_ROOT"

if [ "$_esp" -eq 1 ]; then
    export ESP_ROOT="$_esp_root"
    export CASROOT="$_cas_root"
    export ESP_ARCH=LINUX64
    export CASARCH=.
    _cas_ver="${CASROOT##*/OpenCASCADE-}"
    IFS=. read -r _cas_major _cas_minor _ <<< "$_cas_ver"
    export CASREV="$_cas_major.$_cas_minor"
    unset _cas_ver _cas_major _cas_minor

    # Shared libraries
    _amac_prepend LD_LIBRARY_PATH "$ESP_ROOT/lib"
    _amac_prepend LD_LIBRARY_PATH "$CASROOT/lib"

    # Python bindings — pyESP only, NOT ESP's bundled Python
    export PYTHONPATH="$AMAC_ROOT/src:$AMAC_ROOT:$ESP_ROOT/pyESP:$ESP_ROOT/lib"
else
    # Do not inherit another project's ESP settings.
    unset ESP_ROOT CASROOT ESP_ARCH CASARCH CASREV
fi

# ESP's bundled Python must never be on PATH or LD_LIBRARY_PATH.
_amac_strip PATH "$_esp_dir/Python-"
_amac_strip LD_LIBRARY_PATH "$_esp_dir/Python-"

# The venv's libtbb.so.12 goes first on LD_LIBRARY_PATH (after the ESP prepends and the
# strip above, so it ends up first).
_amac_prepend LD_LIBRARY_PATH "$_venv/lib"

# Suppress user site-packages (blocks conda/pyenv leaks)
export PYTHONNOUSERSITE=1
unset PYTHONHOME

# Put the venv TBB library first. Forcing `tbb` makes a missing or old library
# fail clearly instead of letting Numba choose another threading layer.
export NUMBA_THREADING_LAYER=tbb

# Ensure venv Python is at front after all path manipulations
_amac_prepend PATH "$_venv/bin"

# Export for child scripts
export AMAC_ROOT

# ── Warn-only checks ─────────────────────────────────────────────────────
_py_path="$(command -v python 2>/dev/null || true)"
if [ "$_py_path" != "$_venv/bin/python" ]; then
    _amac_warn "python resolves to ${_py_path:-<missing>}, expected $_venv/bin/python"
fi
if [ "$_esp" -eq 1 ] && ! python -c 'import pyEGADS' &>/dev/null; then
    _amac_warn "pyEGADS is not importable — re-run: bash setup.sh --esp"
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo "AMAC environment activated"
echo "  AMAC_ROOT:    $AMAC_ROOT"
echo "  Python:       $(python --version 2>/dev/null) at $(command -v python 2>/dev/null)"
if [ "$_esp" -eq 1 ]; then
    echo "  ESP_ROOT:     $ESP_ROOT"
    echo "  CASROOT:      $CASROOT"
else
    echo "  ESP/EGADS:    not installed (bash setup.sh --esp to enable)"
fi
echo "  PYTHONPATH:   $PYTHONPATH"
echo "  Threading:    numba tbb ($_venv/lib/libtbb.so.12, interface $_tbb_ver)"
[ "$_pp_dropped" -eq 0 ] || \
echo "                (previous PYTHONPATH entries removed for this terminal; only these are used)"

unset -f _amac_prepend _amac_strip _amac_warn
unset _venv _tbb_ver _tbb_pin _esp_dir _esp_root _cas_root _esp _pp_dropped _py_path _entries
