#  __init__.py
#
#  Copyright (c) 2025 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import asyncio
import json
import logging
import webbrowser
from asyncio import Future
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import cast, AsyncIterator, Final, Any
from urllib.parse import urlencode, quote, unquote_plus

from fastapi import FastAPI, Request, HTTPException
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field
from starlette.datastructures import QueryParams
from uvicorn import Config, Server
from uvicorn.config import LOGGING_CONFIG

BASE_URL = "bear://x-callback-url"

LOGGER = logging.getLogger(__name__)


@dataclass
class ErrorResponse(Exception):
    errorCode: int
    errorMessage: str

    def __str__(self) -> str:
        return self.errorMessage


@dataclass
class AppContext:
    futures: dict[str, Future[QueryParams]]


@asynccontextmanager
async def app_lifespan(_server: FastMCP, uds: Path) -> AsyncIterator[AppContext]:
    callback = FastAPI()
    futures: dict[str, Future[QueryParams]] = {}

    @callback.post("/{req_id}/success", status_code=HTTPStatus.NO_CONTENT, include_in_schema=False)
    def success(req_id: str, req: Request) -> None:
        if req_id not in futures:
            raise HTTPException(status_code=404, detail="Request not found")

        futures[req_id].set_result(req.query_params)

    @callback.post("/{req_id}/error", status_code=HTTPStatus.NO_CONTENT, include_in_schema=False)
    def error(req_id: str, req: Request) -> None:
        if req_id not in futures:
            raise HTTPException(status_code=404, detail="Request not found")

        q = req.query_params
        futures[req_id].set_exception(
            ErrorResponse(
                errorCode=int(q.get("error-Code") or "0"),
                errorMessage=q.get("errorMessage") or "",
            )
        )

    log_config = deepcopy(LOGGING_CONFIG)
    log_config["handlers"]["access"]["stream"] = "ext://sys.stderr"
    server = Server(
        Config(
            app=callback,
            uds=str(uds),
            log_level="warning",
            log_config=log_config,
            h11_max_incomplete_event_size=1024 * 1024,  # 1MB
        )
    )

    LOGGER.info(f"Starting callback server on {uds}")
    server_task = asyncio.create_task(server.serve())
    try:
        yield AppContext(futures=futures)
    finally:
        LOGGER.info("Stopping callback server")
        server.should_exit = True
        await server_task


def server(token: str, uds: Path) -> FastMCP:
    mcp = FastMCP("Bear", lifespan=partial(app_lifespan, uds=uds))

    @mcp.tool()
    async def open_note(
        ctx: Context[Any, AppContext],
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title", default=None),
    ) -> str:
        """Open a note identified by its title or id and return its content."""
        req_id = ctx.request_id
        params = {
            "new_window": "no",
            "float": "no",
            "show_window": "no",
            "open_note": "no",
            "selected": "no",
            "pin": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if id is not None:
            params["id"] = id
        if title is not None:
            params["title"] = title

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/open-note?{urlencode(params, quote_via=quote)}")
            res = await future
            return unquote_plus(res.get("note") or "")

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def create(
        ctx: Context[Any, AppContext],
        title: str | None = Field(description="note title", default=None),
        text: str | None = Field(description="note body", default=None),
        tags: list[str] | None = Field(description="list of tags", default=None),
        timestamp: bool = Field(description="prepend the current date and time to the text", default=False),
    ) -> str:
        """Create a new note and return its unique identifier. Empty notes are not allowed."""
        req_id = ctx.request_id
        params = {
            "open_note": "no",
            "new_window": "no",
            "float": "no",
            "show_window": "no",
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if title is not None:
            params["title"] = title
        if text is not None:
            if title:
                # remove the title from the note text to avoid being duplicated
                text = text.removeprefix("# " + title)
            params["text"] = text
        if tags is not None:
            params["tags"] = ",".join(tags)
        if timestamp:
            params["timestamp"] = "yes"

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/create?{urlencode(params, quote_via=quote)}")
            res = await future
            return res.get("identifier") or ""

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def replace_note(
        ctx: Context[Any, AppContext],
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="new title for the note", default=None),
        text: str | None = Field(description="new text to replace note content", default=None),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
        timestamp: bool = Field(description="prepend the current date and time to the text", default=False),
    ) -> str:
        """Replace the content of an existing note identified by its id."""
        req_id = ctx.request_id
        mode = "replace_all" if title is not None else "replace"
        params = {
            "mode": mode,
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if id is not None:
            params["id"] = id
        if text is not None:
            params["text"] = text
        if title is not None:
            params["title"] = title
        if tags is not None:
            params["tags"] = ",".join(tags)
        if timestamp:
            params["timestamp"] = "yes"

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
            res = await future
            return unquote_plus(res.get("note") or "")

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def tags(
        ctx: Context[Any, AppContext],
    ) -> list[str]:
        """Return all the tags currently displayed in Bear’s sidebar."""
        req_id = ctx.request_id
        params = {
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/tags?{urlencode(params, quote_via=quote)}")
            res = await future

            raw_tags = res.get("tags")
            if raw_tags is None:
                return []

            notes = cast(list[dict[str, str]], json.loads(raw_tags))
            return [note["name"] for note in notes if "name" in note]

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def open_tag(
        ctx: Context[Any, AppContext],
        name: str = Field(description="tag name or a list of tags divided by comma"),
    ) -> list[str]:
        """Show all the notes which have a selected tag in bear."""
        req_id = ctx.request_id
        params = {
            "name": name,
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/open-tag?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def untagged(
        ctx: Context[Any, AppContext],
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Untagged sidebar item."""
        req_id = ctx.request_id
        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if search is not None:
            params["search"] = search

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/untagged?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def todo(
        ctx: Context[Any, AppContext],
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Todo sidebar item."""
        req_id = ctx.request_id
        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if search is not None:
            params["search"] = search

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/todo?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def today(
        ctx: Context[Any, AppContext],
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Today sidebar item."""
        req_id = ctx.request_id
        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if search is not None:
            params["search"] = search

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/today?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def locked(
        ctx: Context[Any, AppContext],
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Locked sidebar item."""
        req_id = ctx.request_id
        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if search is not None:
            params["search"] = search

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/locked?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def search(
        ctx: Context[Any, AppContext],
        term: str | None = Field(description="string to search", default=None),
        tag: str | None = Field(description="tag to search into", default=None),
    ) -> list[str]:
        """Show search results in Bear for all notes or for a specific tag."""
        req_id = ctx.request_id
        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if term is not None:
            params["term"] = term
        if tag is not None:
            params["tag"] = tag

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/search?{urlencode(params, quote_via=quote)}")
            res = await future
            return format_notes(res.get("notes"))

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    @mcp.tool()
    async def grab_url(
        ctx: Context[Any, AppContext],
        url: str = Field(description="url to grab"),
        tags: list[str] | None = Field(
            description="list of tags. If tags are specified in the Bear’s web content preferences, this parameter is ignored.",
            default=None,
        ),
    ) -> str:
        """Create a new note with the content of a web page and return its unique identifier."""
        req_id = ctx.request_id
        params = {
            "url": url,
            "x-success": f"xfwder://{uds.stem}/{req_id}/success",
            "x-error": f"xfwder://{uds.stem}/{req_id}/error",
        }
        if tags is not None:
            params["tags"] = ",".join(tags)

        future = Future[QueryParams]()
        ctx.request_context.lifespan_context.futures[req_id] = future
        try:
            webbrowser.open(f"{BASE_URL}/grab-url?{urlencode(params, quote_via=quote)}")
            res = await future

            return res.get("identifier") or ""

        finally:
            del ctx.request_context.lifespan_context.futures[req_id]

    return mcp


def format_notes(raw: str | None) -> list[str]:
    if raw is None:
        return []

    notes = cast(list[dict], json.loads(raw))
    return [f"{note.get('title')} (ID: {note.get('identifier')})" for note in notes]


__all__: Final = ["server"]
