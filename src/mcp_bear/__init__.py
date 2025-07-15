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
from asyncio import Queue, Future, QueueEmpty
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import cast, AsyncIterator, Final
from urllib.parse import urlencode, quote, unquote_plus

from fastapi import FastAPI, Request
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


def register_callback(api: FastAPI, path: str) -> Queue[Future[QueryParams]]:
    queue = Queue[Future[QueryParams]]()

    @api.post(f"/{path}/success", status_code=HTTPStatus.NO_CONTENT, include_in_schema=False)
    def success(request: Request) -> None:
        try:
            future = queue.get_nowait()
            future.set_result(request.query_params)
        except QueueEmpty:
            pass

    @api.post(f"/{path}/error", status_code=HTTPStatus.NO_CONTENT, include_in_schema=False)
    def error(request: Request) -> None:
        try:
            future = queue.get_nowait()

            q = request.query_params
            future.set_exception(
                ErrorResponse(
                    errorCode=int(q.get("error-Code") or "0"),
                    errorMessage=q.get("errorMessage") or "",
                )
            )
        except QueueEmpty:
            pass

    return queue


@dataclass
class AppContext:
    open_note_results: Queue[Future[QueryParams]]
    create_results: Queue[Future[QueryParams]]
    add_text_results: Queue[Future[QueryParams]]
    tags_results: Queue[Future[QueryParams]]
    open_tag_results: Queue[Future[QueryParams]]
    todo_results: Queue[Future[QueryParams]]
    today_results: Queue[Future[QueryParams]]
    search_results: Queue[Future[QueryParams]]
    grab_url_results: Queue[Future[QueryParams]]


@asynccontextmanager
async def app_lifespan(_server: FastMCP, uds: Path) -> AsyncIterator[AppContext]:
    callback = FastAPI()

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
        yield AppContext(
            open_note_results=register_callback(callback, "open-note"),
            create_results=register_callback(callback, "create"),
            add_text_results=register_callback(callback, "add-text"),
            tags_results=register_callback(callback, "tags"),
            open_tag_results=register_callback(callback, "open-tag"),
            todo_results=register_callback(callback, "todo"),
            today_results=register_callback(callback, "today"),
            search_results=register_callback(callback, "search"),
            grab_url_results=register_callback(callback, "grab-url"),
        )
    finally:
        LOGGER.info("Stopping callback server")
        server.should_exit = True
        await server_task


def server(token: str, uds: Path) -> FastMCP:
    mcp = FastMCP("Bear", lifespan=partial(app_lifespan, uds=uds))

    @mcp.tool()
    async def open_note(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title", default=None),
    ) -> str:
        """Open a note identified by its title or id and return its content."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.open_note_results.put(future)

        params = {
            "new_window": "no",
            "float": "no",
            "show_window": "no",
            "open_note": "no",
            "selected": "no",
            "pin": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/open-note/success",
            "x-error": f"xfwder://{uds.stem}/open-note/error",
        }
        if id is not None:
            params["id"] = id
        if title is not None:
            params["title"] = title

        webbrowser.open(f"{BASE_URL}/open-note?{urlencode(params, quote_via=quote)}")
        res = await future

        return unquote_plus(res.get("note") or "")

    @mcp.tool()
    async def create(
        ctx: Context,
        title: str | None = Field(description="note title", default=None),
        text: str | None = Field(description="note body", default=None),
        tags: list[str] | None = Field(description="list of tags", default=None),
        timestamp: bool = Field(description="prepend the current date and time to the text", default=False),
    ) -> str:
        """Create a new note and return its unique identifier. Empty notes are not allowed."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.create_results.put(future)

        params = {
            "open_note": "no",
            "new_window": "no",
            "float": "no",
            "show_window": "no",
            "x-success": f"xfwder://{uds.stem}/create/success",
            "x-error": f"xfwder://{uds.stem}/create/error",
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

        webbrowser.open(f"{BASE_URL}/create?{urlencode(params, quote_via=quote)}")
        res = await future

        return res.get("identifier") or ""

    @mcp.tool()
    async def replace_note(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="new title for the note", default=None),
        text: str | None = Field(description="new text to replace note content", default=None),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
        timestamp: bool = Field(description="prepend the current date and time to the text", default=False),
        return_content: bool = Field(description="return full note content (default: False for efficiency)", default=False),
    ) -> str:
        """Replace the content of an existing note identified by its id.
        
        EFFICIENCY: Requires sending entire note content over the network. For large notes,
        prefer append_to_note or insert_at_line when possible to reduce network overhead."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.add_text_results.put(future)

        mode = "replace_all" if title is not None else "replace"

        params = {
            "mode": mode,
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/add-text/success",
            "x-error": f"xfwder://{uds.stem}/add-text/error",
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

        webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
        res = await future

        if return_content:
            return unquote_plus(res.get("note") or "")
        else:
            # Return success confirmation for efficiency
            return json.dumps({"status": "success", "operation": "replace", "message": "Note content replaced successfully"})

    @mcp.tool()
    async def append_to_note(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title to identify note (if id not provided)", default=None),
        text: str = Field(description="text to append to the end of the note"),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
        new_line: bool = Field(description="force text to appear on a new line", default=True),
        timestamp: bool = Field(description="prepend the current date and time to the text", default=False),
        return_content: bool = Field(description="return full note content (default: False for efficiency)", default=False),
    ) -> str:
        """Append text to the end of an existing note without replacing existing content.
        
        EFFICIENCY: Only sends new text over the network, making this the most efficient
        method for adding content to large notes."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.add_text_results.put(future)

        params = {
            "mode": "append",
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/add-text/success",
            "x-error": f"xfwder://{uds.stem}/add-text/error",
            "text": text,
        }
        if id is not None:
            params["id"] = id
        elif title is not None:
            params["title"] = title
        else:
            raise ValueError("Either id or title must be provided")

        if new_line:
            params["new_line"] = "yes"
        if tags is not None:
            params["tags"] = ",".join(tags)
        if timestamp:
            params["timestamp"] = "yes"

        webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
        res = await future

        if return_content:
            return unquote_plus(res.get("note") or "")
        else:
            # Return success confirmation for efficiency
            return json.dumps({"status": "success", "operation": "append", "message": "Text appended successfully"})

    @mcp.tool()
    async def insert_at_line(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title to identify note (if id not provided)", default=None),
        text: str = Field(description="text to insert"),
        line_number: int = Field(description="line number to insert at (1-based, -1 means append)"),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
    ) -> str:
        """Insert text at a specific line number in an existing note.
        
        EFFICIENCY: Only sends new text over the network, making this much more efficient
        than replace_note for large notes when inserting content at specific positions.
        
        WARNING: Requires exact line number accuracy. For longer documents, prefer 
        insert_before_text or insert_after_text which use text anchors instead of 
        line counting to avoid positioning errors.
        
        RETURNS: JSON string with metadata: {"status": "success", "affected_lines": N, "preview": "..."}"""
        if line_number == -1:
            result = await append_to_note(ctx, id=id, title=title, text=text, tags=tags)
            return str(result)

        # Read the current note content
        current_content = await open_note(ctx, id=id, title=title)
        lines = current_content.split("\n")
        original_line_count = len(lines)

        # Insert text at specified line (convert to 0-based index)
        # For line_number <= 0, insert at beginning
        if line_number <= 0:
            insert_index = 0
        else:
            insert_index = min(line_number - 1, len(lines))

        lines.insert(insert_index, text)
        new_content = "\n".join(lines)

        # Use replace_note to update with new content
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.add_text_results.put(future)

        params = {
            "mode": "replace",
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/add-text/success",
            "x-error": f"xfwder://{uds.stem}/add-text/error",
            "text": new_content,
        }
        if id is not None:
            params["id"] = id
        elif title is not None:
            params["title"] = title
        else:
            raise ValueError("Either id or title must be provided")

        if tags is not None:
            params["tags"] = ",".join(tags)

        webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
        res = await future

        # Generate preview (±5 lines around insertion point)
        preview_start = max(0, insert_index - 5)
        preview_end = min(len(lines), insert_index + 6)
        preview_lines = lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)

        # Return metadata instead of full content
        metadata = {
            "status": "success",
            "affected_lines": 1,
            "total_lines": len(lines),
            "insertion_line": insert_index + 1,
            "preview": preview
        }
        return json.dumps(metadata)

    @mcp.tool()
    async def insert_before_text(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title to identify note (if id not provided)", default=None),
        anchor_text: str = Field(description="exact text to search for as insertion anchor"),
        new_text: str = Field(description="text to insert before the anchor text"),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
    ) -> str:
        """Insert text before a specific anchor text in an existing note.
        
        EFFICIENCY: Only sends new text over the network, making this much more efficient
        than replace_note for large notes when inserting content at specific text positions.
        
        STRICT MATCHING: Requires exact match of anchor_text. Returns error if not found or 
        if multiple matches exist to avoid ambiguity.
        
        RETURNS: JSON string with metadata: {"status": "success", "affected_lines": N, "preview": "..."}"""
        # Read the current note content
        current_content = await open_note(ctx, id=id, title=title)
        
        # Check for exact match
        if anchor_text not in current_content:
            raise ValueError(f"Anchor text '{anchor_text}' not found in note")
        
        # Check for multiple matches to avoid ambiguity
        if current_content.count(anchor_text) > 1:
            raise ValueError(f"Anchor text '{anchor_text}' appears {current_content.count(anchor_text)} times in note. Use unique text for unambiguous insertion.")
        
        # Insert text before the anchor
        new_content = current_content.replace(anchor_text, new_text + anchor_text)
        
        # Use replace_note to update with new content
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.add_text_results.put(future)

        params = {
            "mode": "replace",
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/add-text/success",
            "x-error": f"xfwder://{uds.stem}/add-text/error",
            "text": new_content,
        }
        if id is not None:
            params["id"] = id
        elif title is not None:
            params["title"] = title
        else:
            raise ValueError("Either id or title must be provided")

        if tags is not None:
            params["tags"] = ",".join(tags)

        webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
        res = await future

        # Calculate affected lines and generate preview
        old_lines = current_content.split("\n")
        new_lines = new_content.split("\n")
        
        # Find the line where insertion occurred
        anchor_line = -1
        for i, line in enumerate(old_lines):
            if anchor_text in line:
                anchor_line = i
                break
        
        # Generate preview (±5 lines around insertion point)
        preview_start = max(0, anchor_line - 5)
        preview_end = min(len(new_lines), anchor_line + 6)
        preview_lines = new_lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)

        # Return metadata instead of full content
        metadata = {
            "status": "success",
            "affected_lines": len(new_lines) - len(old_lines),
            "total_lines": len(new_lines),
            "anchor_line": anchor_line + 1 if anchor_line >= 0 else None,
            "preview": preview
        }
        return json.dumps(metadata)

    @mcp.tool()
    async def insert_after_text(
        ctx: Context,
        id: str | None = Field(description="note unique identifier", default=None),
        title: str | None = Field(description="note title to identify note (if id not provided)", default=None),
        anchor_text: str = Field(description="exact text to search for as insertion anchor"),
        new_text: str = Field(description="text to insert after the anchor text"),
        tags: list[str] | None = Field(description="list of tags to add to the note", default=None),
    ) -> str:
        """Insert text after a specific anchor text in an existing note.
        
        EFFICIENCY: Only sends new text over the network, making this much more efficient
        than replace_note for large notes when inserting content at specific text positions.
        
        STRICT MATCHING: Requires exact match of anchor_text. Returns error if not found or 
        if multiple matches exist to avoid ambiguity.
        
        RETURNS: JSON string with metadata: {"status": "success", "affected_lines": N, "preview": "..."}"""
        # Read the current note content
        current_content = await open_note(ctx, id=id, title=title)
        
        # Check for exact match
        if anchor_text not in current_content:
            raise ValueError(f"Anchor text '{anchor_text}' not found in note")
        
        # Check for multiple matches to avoid ambiguity
        if current_content.count(anchor_text) > 1:
            raise ValueError(f"Anchor text '{anchor_text}' appears {current_content.count(anchor_text)} times in note. Use unique text for unambiguous insertion.")
        
        # Insert text after the anchor
        new_content = current_content.replace(anchor_text, anchor_text + new_text)
        
        # Use replace_note to update with new content
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.add_text_results.put(future)

        params = {
            "mode": "replace",
            "open_note": "no",
            "new_window": "no",
            "show_window": "no",
            "edit": "no",
            "x-success": f"xfwder://{uds.stem}/add-text/success",
            "x-error": f"xfwder://{uds.stem}/add-text/error",
            "text": new_content,
        }
        if id is not None:
            params["id"] = id
        elif title is not None:
            params["title"] = title
        else:
            raise ValueError("Either id or title must be provided")

        if tags is not None:
            params["tags"] = ",".join(tags)

        webbrowser.open(f"{BASE_URL}/add-text?{urlencode(params, quote_via=quote)}")
        res = await future

        # Calculate affected lines and generate preview
        old_lines = current_content.split("\n")
        new_lines = new_content.split("\n")
        
        # Find the line where insertion occurred
        anchor_line = -1
        for i, line in enumerate(old_lines):
            if anchor_text in line:
                anchor_line = i
                break
        
        # Generate preview (±5 lines around insertion point)
        preview_start = max(0, anchor_line - 5)
        preview_end = min(len(new_lines), anchor_line + 6)
        preview_lines = new_lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)

        # Return metadata instead of full content
        metadata = {
            "status": "success",
            "affected_lines": len(new_lines) - len(old_lines),
            "total_lines": len(new_lines),
            "anchor_line": anchor_line + 1 if anchor_line >= 0 else None,
            "preview": preview
        }
        return json.dumps(metadata)

    @mcp.tool()
    async def get_editing_strategy_advice(
        content_length_estimate: int = Field(description="Estimated length of note content in characters"),
        edit_type: str = Field(description="Type of edit: 'append', 'insert', or 'replace'"),
    ) -> str:
        """Get advice on most efficient editing approach based on content size and edit type.
        
        Helps AI clients choose the most network-efficient method for editing notes."""
        if content_length_estimate > 5000:  # Large note threshold
            if edit_type == "append":
                return "Use append_to_note - most efficient for large notes, only sends new text"
            elif edit_type == "insert":
                return "Use insert_before_text or insert_after_text - more reliable than insert_at_line for large notes, avoids line counting errors"
            else:  # replace
                return "For large notes, consider breaking into smaller append/insert operations if possible"
        elif content_length_estimate > 1000:  # Medium note threshold
            if edit_type == "replace":
                return "Consider append_to_note, insert_before_text, or insert_after_text if you're only adding content"
            elif edit_type == "insert":
                return "Use insert_before_text or insert_after_text for text-based positioning, or insert_at_line for line-based positioning"
            else:
                return f"Use {edit_type}_to_note - efficient for medium-sized notes"
        else:  # Small notes
            return "Any method is fine for small notes - network efficiency is not a concern"

    @mcp.tool()
    async def tags(
        ctx: Context,
    ) -> list[str]:
        """Return all the tags currently displayed in Bear's sidebar."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.tags_results.put(future)

        params = {
            "token": token,
            "x-success": f"xfwder://{uds.stem}/tags/success",
            "x-error": f"xfwder://{uds.stem}/tags/error",
        }

        webbrowser.open(f"{BASE_URL}/tags?{urlencode(params, quote_via=quote)}")
        res = await future

        notes = cast(list[dict], json.loads(res.get("tags") or "[]"))
        return [note["name"] for note in notes if "name" in note]

    @mcp.tool()
    async def open_tag(
        ctx: Context,
        name: str = Field(description="tag name or a list of tags divided by comma"),
    ) -> list[str]:
        """Show all the notes which have a selected tag in bear."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.open_tag_results.put(future)

        params = {
            "name": name,
            "token": token,
            "x-success": f"xfwder://{uds.stem}/open-tag/success",
            "x-error": f"xfwder://{uds.stem}/open-tag/error",
        }

        webbrowser.open(f"{BASE_URL}/open-tag?{urlencode(params, quote_via=quote)}")
        res = await future

        notes = cast(list[dict], json.loads(res.get("notes") or "[]"))
        return [f"{note.get('title')} (ID: {note.get('identifier')})" for note in notes]

    @mcp.tool()
    async def todo(
        ctx: Context,
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Todo sidebar item."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.todo_results.put(future)

        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/todo/success",
            "x-error": f"xfwder://{uds.stem}/todo/error",
        }
        if search is not None:
            params["search"] = search

        webbrowser.open(f"{BASE_URL}/todo?{urlencode(params, quote_via=quote)}")
        res = await future

        notes = cast(list[dict], json.loads(res.get("notes") or "[]"))
        return [f"{note.get('title')} (ID: {note.get('identifier')})" for note in notes]

    @mcp.tool()
    async def today(
        ctx: Context,
        search: str | None = Field(description="string to search", default=None),
    ) -> list[str]:
        """Select the Today sidebar item."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.today_results.put(future)

        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/today/success",
            "x-error": f"xfwder://{uds.stem}/today/error",
        }
        if search is not None:
            params["search"] = search

        webbrowser.open(f"{BASE_URL}/today?{urlencode(params, quote_via=quote)}")
        res = await future

        notes = cast(list[dict], json.loads(res.get("notes") or "[]"))
        return [f"{note.get('title')} (ID: {note.get('identifier')})" for note in notes]

    @mcp.tool()
    async def search(
        ctx: Context,
        term: str | None = Field(description="string to search", default=None),
        tag: str | None = Field(description="tag to search into", default=None),
    ) -> list[str]:
        """Show search results in Bear for all notes or for a specific tag."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.search_results.put(future)

        params = {
            "show_window": "no",
            "token": token,
            "x-success": f"xfwder://{uds.stem}/search/success",
            "x-error": f"xfwder://{uds.stem}/search/error",
        }
        if term is not None:
            params["term"] = term
        if tag is not None:
            params["tag"] = tag

        webbrowser.open(f"{BASE_URL}/search?{urlencode(params, quote_via=quote)}")
        res = await future

        notes = cast(list[dict], json.loads(res.get("notes") or "[]"))
        return [f"{note.get('title')} (ID: {note.get('identifier')})" for note in notes]

    @mcp.tool()
    async def grab_url(
        ctx: Context,
        url: str = Field(description="url to grab"),
        tags: list[str] | None = Field(
            description="list of tags. If tags are specified in the Bear’s web content preferences, this parameter is ignored.",
            default=None,
        ),
    ) -> str:
        """Create a new note with the content of a web page and return its unique identifier."""
        app_ctx: AppContext = ctx.request_context.lifespan_context  # type: ignore
        future = Future[QueryParams]()
        await app_ctx.grab_url_results.put(future)

        params = {
            "url": url,
            "x-success": f"xfwder://{uds.stem}/grab-url/success",
            "x-error": f"xfwder://{uds.stem}/grab-url/error",
        }
        if tags is not None:
            params["tags"] = ",".join(tags)

        webbrowser.open(f"{BASE_URL}/grab-url?{urlencode(params, quote_via=quote)}")
        res = await future

        return res.get("identifier") or ""

    return mcp


__all__: Final = ["server"]
