"""Minimal local replacement for the pyext RuntimeModule API used by LCB."""

import types


class RuntimeModule:
    """Load Python source into an isolated module object."""

    @staticmethod
    def from_string(name, path, source):
        module = types.ModuleType(name)
        module.__dict__["__name__"] = name
        exec(compile(source, path or f"<{name}>", "exec"), module.__dict__)
        return module
