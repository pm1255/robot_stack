import locale
from pathlib import Path
import tempfile
import unittest
from robot_stack.inventory import file_tasks


class InventoryContracts(unittest.TestCase):
    def test_python_sources_use_declared_encoding_after_native_locale_change(self):
        previous = locale.setlocale(locale.LC_CTYPE)
        try:
            with tempfile.TemporaryDirectory() as root:
                folder=Path(root)
                (folder/'task.py').write_text('# 中文任务注释\nclass task: pass\n',encoding='utf-8')
                (folder/'helper.py').write_text('class Other: pass\n',encoding='utf-8')
                (folder/'_base.py').write_text('class _base: pass\n',encoding='utf-8')
                locale.setlocale(locale.LC_CTYPE,'C')
                self.assertEqual(file_tasks(folder),['task'])
        finally:
            locale.setlocale(locale.LC_CTYPE,previous)
