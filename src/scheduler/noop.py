# src/scheduler/noop.py

class NoOpScheduler:
    """
    Placeholder scheduler that does nothing.
    This keeps training code simple:
        scheduler.step(...)
    works regardless of scheduler type.
    """

    def step(self, *args, **kwargs):
        return None
