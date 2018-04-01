import unittest

class PillowTestCase(unittest.TestCase):
    def assert_image_equal(self, a, b, msg=None):
        self.assertEqual(
            a.mode, b.mode,
            msg or "got mode %r, expected %r" % (a.mode, b.mode))
        self.assertEqual(
            a.size, b.size,
            msg or "got size %r, expected %r" % (a.size, b.size))
        if a.tobytes() != b.tobytes():
            self.fail(msg or "got different content")
