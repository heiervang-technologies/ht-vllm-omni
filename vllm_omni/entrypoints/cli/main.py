"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import sys


def _first_positional_argument(argv: list[str]) -> str | None:
    return next((argument for argument in argv[1:] if not argument.startswith("-")), None)


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    # Check if --omni flag is present
    if "--omni" not in sys.argv:
        from vllm.entrypoints.cli.main import main as vllm_main

        vllm_main()
        return
    else:
        # Force colored logging even when piped (e.g. `| tee`).
        # Must be set before any vLLM import because the logger
        # formatter is configured at import time via _use_color().
        import os

        if "VLLM_LOGGING_COLOR" not in os.environ:
            os.environ["VLLM_LOGGING_COLOR"] = "1"

        from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        selected_command = _first_positional_argument(sys.argv)
        command_modules = []
        if selected_command != "bench":
            import vllm_omni.entrypoints.cli.serve

            command_modules.append(vllm_omni.entrypoints.cli.serve)
        if selected_command != "serve":
            import vllm_omni.entrypoints.cli.benchmark.main

            if selected_command == "bench":
                # Apply benchmark patches before registering the concrete
                # benchmark type. Root help only needs the lightweight parent.
                importlib.import_module("vllm_omni.benchmarks.patch")
                import vllm_omni.entrypoints.cli.benchmark.serve  # noqa: F401

            command_modules.append(vllm_omni.entrypoints.cli.benchmark.main)

        cli_env_setup()

        from vllm_omni.entrypoints.cli.serve import _ensure_vllm_platform

        _ensure_vllm_platform()

        parser = FlexibleArgumentParser(
            description="vLLM OMNI CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm_omni"),
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in command_modules:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
