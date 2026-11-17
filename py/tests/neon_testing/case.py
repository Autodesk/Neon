"""Base TestCase with deterministic teardown of Neon objects."""

import gc
import unittest

import warp as wp

import neon

from .runtime import init_once, run_container


def _release(resources) -> None:
    wp.synchronize()
    while resources:
        obj = resources.pop()
        cleanup = getattr(obj, "cleanup", None)
        if cleanup is not None:
            cleanup()
    gc.collect()


class NeonTestCase(unittest.TestCase):
    """Initializes Warp/Neon once and releases tracked objects in reverse order.

    Track every Backend, grid, field, container and skeleton a test creates
    (``self.track(...)``, ``self.backend()``, ``self.run_op(...)``). They are
    cleaned up after the test, newest first, so the backend always goes last.
    Objects created in ``setUpClass`` go through ``cls.track_class(...)``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        init_once()
        cls._class_resources = []
        cls.addClassCleanup(_release, cls._class_resources)

    def setUp(self):
        super().setUp()
        self._resources = []
        self.addCleanup(_release, self._resources)

    # -- tracking -----------------------------------------------------------

    def track(self, obj):
        self._resources.append(obj)
        return obj

    @classmethod
    def track_class(cls, obj):
        cls._class_resources.append(obj)
        return obj

    # -- factories ----------------------------------------------------------

    @staticmethod
    def _new_backend(devices):
        return neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=list(devices))

    def backend(self, devices=(0,)):
        return self.track(self._new_backend(devices))

    @classmethod
    def class_backend(cls, devices=(0,)):
        return cls.track_class(cls._new_backend(devices))

    def run_op(self, container, stream_idx: int = 0):
        """Track ``container``, run it on the Neon runtime and synchronize."""
        run_container(self.track(container), stream_idx)
        return container
