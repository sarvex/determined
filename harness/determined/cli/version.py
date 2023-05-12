import argparse
import sys
from typing import Any, Dict, List

import requests
import termcolor
from packaging import version

import determined
import determined.cli
from determined.common import api
from determined.common.declarative_argparse import Cmd

from . import render


def get_version(host: str) -> Dict[str, Any]:
    client_info = {"version": determined.__version__}

    master_info = {"cluster_id": "", "master_id": "", "version": ""}

    try:
        master_info = api.get(host, "info", authenticated=False).json()
        # Most connection errors mean that the master is unreachable, which this function handles.
        # An SSL error, however, means it was reachable but something went wrong, so let that error
        # propagate out.
    except requests.exceptions.SSLError:
        raise
    except api.errors.MasterNotFoundException:
        pass

    return {"client": client_info, "master": master_info, "master_address": host}


def check_version(parsed_args: argparse.Namespace) -> None:
    info = get_version(parsed_args.master)

    master_version = info["master"]["version"]
    client_version = info["client"]["version"]
    if not master_version:
        print(
            termcolor.colored(
                f"Master not found at {parsed_args.master}. Hint: Remember to set the DET_MASTER environment variable to the correct Determined master IP or use the '-m' flag.",
                "yellow",
            ),
            file=sys.stderr,
        )
    elif version.Version(client_version) < version.Version(master_version):
        print(
            termcolor.colored(
                f"CLI version {client_version} is less than master version {master_version}. Consider upgrading the CLI.",
                "yellow",
            ),
            file=sys.stderr,
        )
    elif version.Version(client_version) > version.Version(master_version):
        print(
            termcolor.colored(
                f"Master version {master_version} is less than CLI version {client_version}. Consider upgrading the master.",
                "yellow",
            ),
            file=sys.stderr,
        )


def describe_version(parsed_args: argparse.Namespace) -> None:
    info = get_version(parsed_args.master)

    print(render.format_object_as_yaml(info))


args_description = [
    Cmd("version", describe_version, "show version information", [])
]  # type: List[Any]
