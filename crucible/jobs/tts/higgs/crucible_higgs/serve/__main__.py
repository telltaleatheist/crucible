"""`python -m crucible_higgs.serve` - the resident Higgs v3 narration worker.

The entry point `crucible/engines/narrator.py` spawns, in the tts env's
interpreter, with `PYTHONPATH` naming the directory this package sits in. The
worker's whole interface is the JSON-lines protocol on stdin/stdout (see
worker.py) and its configuration is the env that adapter writes. The one
command-line argument is `--fake-engine`, the protocol-test door; it is parsed in
worker.main() and anything else is a hard error.
"""
import sys

from .worker import main

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
