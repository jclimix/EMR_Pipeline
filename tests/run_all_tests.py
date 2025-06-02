import os
import pytest

TEST_DIR = "tests"

if __name__ == "__main__":
    for filename in os.listdir(TEST_DIR):
        if filename.startswith("test_") and filename.endswith(".py"):
            filepath = os.path.join(TEST_DIR, filename)
            pytest.main(["-v", filepath])
