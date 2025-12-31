"""Oneiro Discord Bot - Entry point."""

import os

from oneiro.bot import create_bot


def main() -> None:
    """Run the Oneiro Discord bot."""
    if os.environ.get("ENABLE_PROGRESS_BARS", "").lower() not in ("1", "true", "yes", "on"):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.disable_progress_bar()

    token = os.environ.get("TOKEN")
    if not token:
        raise ValueError("TOKEN environment variable required")

    bot = create_bot()
    bot.run(token)


if __name__ == "__main__":
    main()
