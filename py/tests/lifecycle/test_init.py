import unittest
from unittest import mock

from neon_testing import NeonTestCase

import warp as wp
import neon


class TestNeonInit(NeonTestCase):
    def test_initialized_after_setup(self):
        self.assertTrue(neon.is_initialized())

    def test_second_init_keeps_kernel_cache(self):
        with mock.patch.object(wp, "clear_kernel_cache") as clear:
            neon.init()
            neon.init(clear_kernel_cache=False)
        clear.assert_not_called()
        self.assertTrue(neon.is_initialized())

    def test_explicit_clear_after_init(self):
        with mock.patch.object(wp, "clear_kernel_cache") as clear:
            neon.init(clear_kernel_cache=True)
        clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
