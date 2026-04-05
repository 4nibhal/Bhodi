import io
from contextlib import redirect_stdout
from unittest import TestCase
from unittest.mock import patch

import indexer.cli as cli


class CliCompatibilityTest(TestCase):
    def test_main_delegates_to_platform_cli(self) -> None:
        stdout = io.StringIO()

        with patch.object(cli, "platform_main") as platform_main:
            with redirect_stdout(stdout):
                cli.main()

        platform_main.assert_called_once_with()
