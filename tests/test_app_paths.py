from __future__ import annotations

import unittest
from pathlib import Path

from app import resolve_app_root


class AppPathTest(unittest.TestCase):
    def test_relative_script_path_resolves_to_absolute_app_directory(self) -> None:
        root = resolve_app_root("app.py")

        self.assertTrue(root.is_absolute())
        self.assertEqual(root, Path.cwd().resolve())


if __name__ == "__main__":
    unittest.main()
