# amac-3d

amac-3d adapts 3D tetrahedral meshes to an anisotropic metric field. It splits,
collapses, swaps and smooths edges, and can project boundaries onto STEP geometry
through optional ESP/EGADS support.

## Install

On Linux x86-64 with CPython 3.14, run from the repository root:

```bash
bash setup.sh
source activate.sh
```

For STEP geometry, use `bash setup.sh --esp` before activation. Setup installs the
pinned dependencies from `requirements.txt` into `.venv`.

## Quick start

```bash
python -m run.adapt cases/cube_cross/case.toml --check
python -m run.adapt cases/cube_cross/case.toml --iterations 12 --threads 4 --output-dir /tmp/amac-cube-cross
```

The run writes its adapted mesh, history and log in the chosen directory. See the
[user guide](docs/amac_3d_guide.pdf) for cases and commands, and the
[code and reference manuals](docs/%6Clm-guides/) for engine and format details.

## Third-party material

Dependencies install as separate packages under their own licenses; optional ESP
comes from the ESP distribution. The bundled Netgen meshes are generated inputs.
The code follows published algorithms, including NASA refine and work by Loseille
and Alauzet, Loseille and Löhner, Knupp, and Vlachos et al. These are references
for the methods; no refine source code is bundled.

## License

LGPL-2.1-only. See [LICENSE](LICENSE).
