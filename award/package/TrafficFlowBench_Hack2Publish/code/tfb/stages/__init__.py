"""Vendored Kaggle kernels, one module per stage. Each module is the original kernel source (consolidation/lineage_src/)
with only its I/O and recipe-constant lines changed; every changed or added line ends in a '[TFB]' comment and
tests/test_static.py enforces that. The modules run top to bottom as __main__ inside tfb.stage_exec (never import them)."""
