import asyncio
import sys


def asyncio_setup() -> None:
    # Set eventloop for win32 setups

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy)
