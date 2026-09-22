"""Triage one support ticket with eight typed decisions and a single prefill.

    uv run examples/triage.py
"""

from jeff import Jeff, boolean, choice, score

TICKET = """
From: priya@northwind.example
Subject: still charged after cancelling

I cancelled the Team plan on 2 February, well before the renewal, and I have the
confirmation email. Today I was charged EUR 480 anyway. This is the second time
this has happened to us. I need the charge reversed today or we will dispute it
with our bank.
"""

QUESTIONS = [
    boolean("Is the customer asking for a refund?", id="wants_refund"),
    boolean("Is the customer threatening a chargeback?", id="chargeback_risk"),
    boolean("Has this problem happened to them before?", id="repeat_issue"),
    boolean("Is the customer angry?", id="angry"),
    boolean("Does this need a human, rather than an automated reply?", id="needs_human"),
    choice("Which queue should this go to?", ["billing", "technical", "sales", "abuse"], id="queue"),
    choice("What is the root cause?", ["failed cancellation", "duplicate charge", "user error", "unclear"], id="cause"),
    score("How urgent is this, from 1 (whenever) to 5 (right now)?", 1, 5, id="urgency"),
]


def main() -> None:
    jeff = Jeff()
    # Debiasing is off by default; see examples/letter_bias.py for what it buys.
    answers = jeff.ask(TICKET.strip(), QUESTIONS, permutations="all")

    for answer in answers:
        bar = "#" * round(answer.confidence * 20)
        print(f"{answer.question.key:<16} {answer.choice:<20} {answer.confidence:5.1%} {bar}")

    stats = jeff.stats
    print(
        f"\n{stats['questions']} decisions in {stats['total_seconds'] * 1000:.0f} ms "
        f"({stats['output_tokens']} tokens generated)."
    )
    print(
        f"Prefilled {stats['prefix_tokens']} shared tokens once instead of "
        f"{stats['questions']} times: {stats['saved_tokens']} tokens of prefill skipped."
    )
    urgency = next(a for a in answers if a.question.key == "urgency")
    print(f"Expected urgency: {urgency.expected_value:.2f}/5")


if __name__ == "__main__":
    main()
