"""Jeff — typed probabilistic decisions from one shared prefill.

    from jeff import Jeff, boolean, choice, score

    jeff = Jeff()
    answers = jeff.ask(ticket, [
        boolean("Is the customer asking for a refund?"),
        choice("Which queue?", ["billing", "access", "bug report"]),
        score("How urgent is this?", 1, 5),
    ])
"""

from .calibrate import fit_temperature
from .engine import DEFAULT_MODEL, Jeff
from .prompt import Answer, Question, boolean, choice, score

__all__ = [
    "Answer",
    "DEFAULT_MODEL",
    "Jeff",
    "Question",
    "boolean",
    "choice",
    "fit_temperature",
    "score",
]
