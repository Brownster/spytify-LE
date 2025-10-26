import argparse

from .service_app import run_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoti2 service wrapper with web configuration UI",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the web UI (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8730,
        help="Port for the web UI (default: 8730)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the spotify-splitter config file to manage",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_service(host=args.host, port=args.port, config=args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
