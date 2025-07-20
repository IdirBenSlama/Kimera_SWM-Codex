"""Test async context manager"""
import asyncio
from contextlib import asynccontextmanager

class TestClass:
    @asynccontextmanager
    async def test_method(self):
        print("Entering context")
        yield "test_value"
        print("Exiting context")

async def main():
    obj = TestClass()
    async with obj.test_method() as value:
        print(f"Got value: {value}")

if __name__ == "__main__":
    asyncio.run(main())