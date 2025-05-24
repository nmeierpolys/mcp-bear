#  cli.py
#
#  Copyright (c) 2025 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import logging
import random
import string
import subprocess
import sys
import tarfile
from logging import Logger
from pathlib import Path

import requests
import rich_click as click
from requests import HTTPError

from mcp_bear import server


def generate_file_suffix(length: int = 6) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=length))


def init_forwarder(logger: Logger) -> None:
    forwarder_dir = Path.home().joinpath("Library", "Application Support", "xfwder")
    forwarder = forwarder_dir.joinpath("xFwder.app")
    if not forwarder.exists():
        logger.info("xFwder.app doesn't exist, downloading it")
        forwarder_dir.mkdir(parents=True, exist_ok=True)
        temp_path = forwarder_dir.joinpath("XFwder.tar.gz")

        response = requests.get(
            "https://github.com/jkawamoto/xfwder/releases/download/v0.1.0/XFwder.tar.gz", stream=True
        )
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with tarfile.open(temp_path, "r:gz") as tar:
            tar.extractall(forwarder_dir)

        logger.info("Initializing xFwder")
        subprocess.call(["open", forwarder, "--args", "--init"])


@click.command()
@click.option("--token", envvar="BEAR_API_TOKEN", required=True, help="Bear API token")
@click.version_option()
def main(token: str) -> None:
    """A MCP server for interacting with Bear note-taking software."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("Preparing XFwder")
    try:
        init_forwarder(logger)
    except HTTPError as e:
        logger.error("Failed to initialize XFwder", exc_info=e)
        sys.exit(1)

    logger.info("Preparing a UDS")
    while True:
        uds = Path("/tmp").joinpath(f"mcp-bear-{generate_file_suffix()}.sock")
        if not uds.exists():
            break
    mcp = server(token, uds)

    logger.info("Starting MCP server (Press CTRL+D to quit)")
    mcp.run()
    logger.info("MCP server stopped")
