"""Load verified original source bytes without sys.path or bare-module collisions.

No source rewriting, substitute model, or bytecode write in historical directories.
The original class/function bodies execute unchanged.
"""
from __future__ import annotations
import builtins
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

_instances = itertools.count()

class HistoricalSource:
    MODULES = {"utils", "models", "data", "signals_xib", "train_xib_unified"}

    def __init__(self, root, expected_hashes=None):
        self.root = Path(root).resolve()
        if expected_hashes is None:
            expected_hashes = json.loads((Path(__file__).parent / "configs/source_pin.json").read_text())["files"]
        self.hashes = dict(expected_hashes)
        if set(self.hashes) != {n + ".py" for n in self.MODULES}:
            raise ValueError("All five original source files must be pinned")
        self.modules = {}
        self.prefix = "bws_historical_" + self.hashes["signals_xib.py"][:12] + "_" + str(next(_instances))
        self.verify()

    def _read(self, filename):
        p = self.root / filename
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.hashes[filename]:
            raise ValueError("Historical source hash mismatch: " + filename)
        return data

    def verify(self):
        for name in self.hashes:
            self._read(name)

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in self.MODULES:
            return self.load(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, name):
        if name not in self.MODULES:
            raise ValueError("Unexpected historical module: " + name)
        if name in self.modules:
            return self.modules[name]
        filename = name + ".py"
        data = self._read(filename)
        qualified = self.prefix + "_" + name
        spec = importlib.util.spec_from_file_location(qualified, self.root / filename)
        module = importlib.util.module_from_spec(spec)
        module.__builtins__ = dict(vars(builtins), __import__=self._import)
        self.modules[name] = module
        sys.modules[qualified] = module  # dataclasses consult their defining module.
        try:
            exec(compile(data, str(self.root / filename), "exec"), module.__dict__)
        except BaseException:
            self.modules.pop(name, None)
            sys.modules.pop(qualified, None)
            raise
        return module
