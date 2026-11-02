import unittest

import numpy as np

from task import Task


class TestTask(unittest.TestCase):
    def test_work(self):
        # Create a small task
        task = Task(identifier=1, size=10)

        # Solve the system Ax = B
        task.work()

        # Verify that Ax is approximately equal to B
        np.testing.assert_allclose(task.a @ task.x, task.b)

        # Verify that the execution time is positive
        self.assertGreater(task.time, 0)


if __name__ == "__main__":
    unittest.main()
