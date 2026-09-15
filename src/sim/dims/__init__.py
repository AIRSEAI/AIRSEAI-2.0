"""Task dimensions: plain numbers, no imports.

One module per task, holding the sizes the asset generator builds from and the
geometric thresholds the criteria judge against. Import a name from here in
any environment, with or without Isaac::

    from sim.dims.<task> import BOARD_X

After editing a module here, regenerate that task's USDs::

    python benchmark/run/make_<task>_assets.py --headless
"""
