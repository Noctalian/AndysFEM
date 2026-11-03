Rebuilt MAPDL-compatible solver: runs inside fem-client (window: Run, or `fem run`).
The run executes geometry.py (Gmsh), reads cell.msh, writes the APDL deck job_local.dat (N, E, CE, D, SOLVE),
solves it in-process and writes job_local.out and results.json.
