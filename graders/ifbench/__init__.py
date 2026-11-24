"""IFBench's checkers (allenai/IFBench at 488918665d53, Apache-2.0, LICENSE here): the graders of task "ifbench".

Only on evaluation machines, in the benchmark items pack (evaluation/benchdata.py), never on a run's machine: they
generate constraints of exactly the evaluated kinds. Changed from upstream: nothing runs at import (no nltk download,
no data directory created beside the package); the nltk data comes from the setup bundle.
"""
