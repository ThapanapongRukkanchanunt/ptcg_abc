import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class CliLazyImportTests(unittest.TestCase):
    def test_phase5_compare_parser_does_not_import_lxml(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_root = repo_root / "src"
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockLxml(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "lxml" or fullname.startswith("lxml."):
                        raise ModuleNotFoundError("blocked lxml")
                    return None

            sys.meta_path.insert(0, BlockLxml())

            from ptcg_abc.cli import build_parser

            args = build_parser().parse_args(
                [
                    "phase5-compare-benchmarks",
                    "--baseline",
                    "old.json",
                    "--candidate",
                    "new.json",
                ]
            )
            assert args.func.__name__ == "command_phase5_compare_benchmarks"
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
