"""Entry points. Every module here is run, never imported.

Each defines main() behind a `__main__` guard and is reachable as a pixi task.
Nothing outside this package imports from it; a test enforces both rules, after
an unguarded driver once ran its whole benchmark on import and overwrote a
recorded result file.
"""
