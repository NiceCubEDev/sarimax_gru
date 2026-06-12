"""
Small CLI progress indicator for long-running offline jobs.
"""

import sys
import threading
import time
from types import TracebackType


class ProgressSpinner:
    """Render a simple spinner to stderr while keeping stdout clean for JSON."""

    def __init__(self, message: str, interval_seconds: float = 0.1) -> None:
        self.message = message
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._use_animation = sys.stderr.isatty()

    def __enter__(self) -> "ProgressSpinner":
        self._started_at = time.monotonic()
        if not self._use_animation:
            print(f"{self.message}...", file=sys.stderr, flush=True)
            return self

        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        elapsed_seconds = time.monotonic() - self._started_at
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        status = "Ошибка" if exc_type is not None else "Готово"
        final_message = f"{status}: {self.message} за {elapsed_seconds:.1f} сек."

        if self._use_animation:
            sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.write(final_message + "\n")
            sys.stderr.flush()
        else:
            print(final_message, file=sys.stderr, flush=True)

    def _animate(self) -> None:
        frames = "|/-\\"
        frame_index = 0
        while not self._stop_event.is_set():
            elapsed_seconds = time.monotonic() - self._started_at
            frame = frames[frame_index % len(frames)]
            sys.stderr.write(f"\r{frame} {self.message}... {elapsed_seconds:.1f} сек.")
            sys.stderr.flush()
            frame_index += 1
            self._stop_event.wait(self.interval_seconds)
