#  test_server.py
#
#  Copyright (c) 2025 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php

from typing import AsyncGenerator

import pytest
from mcp import StdioServerParameters, ClientSession, stdio_client

params = StdioServerParameters(command="uv", args=["run", "mcp-bear", "--token", "abcdefg"])


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
async def mcp_client_session() -> AsyncGenerator[ClientSession, None]:
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session


@pytest.mark.anyio
async def test_list_tools(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.list_tools()
    tools = set(tool.name for tool in res.tools)

    assert "open_note" in tools
    assert "create" in tools
    assert "tags" in tools
    assert "open_tag" in tools
    assert "todo" in tools
    assert "today" in tools
    assert "search" in tools
    assert "grab_url" in tools
    assert "replace_note" in tools
    assert "append_to_note" in tools
    assert "insert_at_line" in tools
    assert "insert_before_text" in tools
    assert "insert_after_text" in tools
    assert "get_editing_strategy_advice" in tools
