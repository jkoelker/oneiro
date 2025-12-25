"""Oneiro Discord Bot - Entry point."""

import os

from oneiro.bot import create_bot


def main() -> None:
    """Run the Oneiro Discord bot."""
    token = os.environ.get("TOKEN")
    if not token:
        raise ValueError("TOKEN environment variable required")

    bot = create_bot()
    bot.run(token)


if __name__ == "__main__":
    main()
