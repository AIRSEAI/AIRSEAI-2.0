# SPDX-License-Identifier: Apache-2.0
"""Preserve a failing exit status when the simulator installs shutdown hooks."""

import os
import sys
import traceback


def install_error_handler() -> None:
    def fail(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.excepthook = fail
