"""Snake, played one typed decision at a time.

    uv run examples/snake.py
    uv run examples/snake.py --model Qwen/Qwen3-8B --watch   # live view
    uv run examples/snake.py --blind              # the grid alone, no notes
    uv run examples/snake.py --player greedy      # the scripted baselines
    uv run examples/snake.py --no-shield          # let the model's choice stand

Each move asks three yes/no questions per direction — does it survive, does it
get closer to the food, is it a dead end — twelve decisions scored off a single
prefill of the board, each under a full cyclic cover so no answer wins because
of the letter it wore. The policy that turns them into a move is plain Python.
Zero tokens are generated.

By default the state also carries a line of plain fact per direction, the way
omp-laya-judge's snake demo has a deterministic planner describe the moves and
lets the model rank them. That makes the job mostly reading, so compare
`--player greedy`, which acts on the same facts with no model at all. `--blind`
drops the notes and leaves the grid and coordinates: the real spatial question.

What it shows, on Qwen3 0.6B and 1.7B: nothing here plays Snake. With notes the
1.7B reads "the snake dies" reliably and never wants to crash, but answers yes
to every food question however it is worded — including "is this the wrong
way?" — so it survives without ever hunting. Blind, both models want a fatal
move most turns. `--ask menu` is the obvious one-question framing, and it is
worse than useless: the 1.7B picks whichever option's note looks unlike the
others, which here is the fatal one on 95% of moves.

The shield is ordinary Python. If the model's favourite move is fatal, the most
probable *safe* move is played instead, and that override is counted rather than
hidden: it is the honest measure of how often the model wanted to die.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from jeff import Jeff, boolean, choice

Cell = tuple[int, int]  # (row, col)

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# --- the game: nothing here knows a model exists ----------------------------


@dataclass
class Game:
    height: int = 8
    width: int = 8
    seed: int = 0
    body: deque[Cell] = field(default_factory=deque)  # head first
    food: Cell | None = None
    score: int = 0
    alive: bool = True

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        row, col = self.height // 2, self.width // 2
        self.body = deque([(row, col), (row, col - 1), (row, col - 2)])
        self.place_food()

    @property
    def head(self) -> Cell:
        return self.body[0]

    def place_food(self) -> None:
        free = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if (r, c) not in self.body
        ]
        self.food = self.rng.choice(free) if free else None

    def target(self, move: str) -> Cell:
        dr, dc = MOVES[move]
        return self.head[0] + dr, self.head[1] + dc

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def blocked(self, cell: Cell, eating: bool) -> bool:
        """Walls and body are fatal; the tail moves away unless the snake grows."""
        if not self.inside(cell):
            return True
        body = list(self.body) if eating else list(self.body)[:-1]
        return cell in body

    def safe(self, move: str) -> bool:
        cell = self.target(move)
        return not self.blocked(cell, eating=cell == self.food)

    def room(self, move: str) -> int:
        """Cells reachable from the head after this move: a cheap trap detector."""
        if not self.safe(move):
            return 0
        cell = self.target(move)
        eating = cell == self.food
        body = [cell] + list(self.body)[: len(self.body) if eating else -1]
        walls = set(body[1:])
        seen, queue = {cell}, deque([cell])
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES.values():
                nxt = (r + dr, c + dc)
                if self.inside(nxt) and nxt not in walls and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)

    def distance(self, cell: Cell) -> int:
        return abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1])

    def step(self, move: str) -> None:
        cell = self.target(move)
        eating = cell == self.food
        if self.blocked(cell, eating):
            self.alive = False
            return
        self.body.appendleft(cell)
        if eating:
            self.score += 1
            self.place_food()
            if self.food is None:  # the board is full: that is a win
                self.alive = False
        else:
            self.body.pop()

    def grid(self) -> str:
        rows = ["#" * (self.width + 2)]
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                if (r, c) == self.head:
                    line += "H"
                elif (r, c) in self.body:
                    line += "o"
                elif (r, c) == self.food:
                    line += "*"
                else:
                    line += "."
            rows.append(f"#{line}#")
        rows.append(rows[0])
        return "\n".join(rows)


# --- the state the model reads, and the questions it answers ----------------


def describe(game: Game, move: str) -> str:
    """One line of fact about a direction. Facts only: the model does the ranking.

    Distances are given as closer/farther rather than as two numbers, since a
    1.7B cannot compare "9 steps (now 8)". It cannot read closer/farther either,
    but this is the version that at least asks it fairly.
    """
    cell = game.target(move)
    if not game.inside(cell):
        return "a wall, so the snake dies"
    if game.blocked(cell, eating=cell == game.food):
        return "your own body, so the snake dies"
    room = game.room(move)
    space = (f"a dead end with only {room} cells of room" if room < len(game.body)
             else f"open space ({room} cells of room)")
    if cell == game.food:
        return f"eats the food; {space}"
    step = game.distance(cell) - game.distance(game.head)
    toward = "one step closer to the food" if step < 0 else "one step farther from the food"
    return f"an empty cell, {toward}; {space}"


def state(game: Game, blind: bool = False) -> str:
    """Everything shared by this move's questions, so it is prefilled once."""
    text = (
        "You are the snake in a game of Snake. Eat the food (*) to grow. Hitting a "
        "wall (#) or your own body (o) ends the game. Rows count down from 0 at the "
        f"top, columns right from 0 at the left.\n\n{game.grid()}\n\n"
        f"Head (H) at row {game.head[0]}, column {game.head[1]}. "
        f"Food at row {game.food[0]}, column {game.food[1]}. Length {len(game.body)}."
    )
    if not blind:
        text += "\n\n" + "\n".join(f"Moving {m}: {describe(game, m)}." for m in MOVES)
    return text


def menu(game: Game, blind: bool):
    """The obvious framing, one choice over four moves. It does not work: see the top."""
    return [choice(
        "Which way should the snake move next?",
        list(MOVES),
        id="move",
        descriptions=None if blind else [describe(game, m) for m in MOVES],
    )]


def each(game: Game, blind: bool):
    """Three yes/no questions per direction, twelve in all, off one prefill."""
    return [
        q for m in MOVES for q in (
            boolean(f"If the snake moves {m}, does it survive?", id=f"safe_{m}"),
            boolean(f"Does moving {m} bring the snake closer to the food?", id=f"food_{m}"),
            boolean(f"Does moving {m} lead into a dead end?", id=f"trap_{m}"),
        )
    ]


def preference(answers: dict) -> dict[str, float]:
    """The policy, in Python: live first, then no trap, then toward the food."""
    if "move" in answers:
        return answers["move"].distribution
    scores = {
        m: answers[f"safe_{m}"]["yes"]
        * (1.0 - 0.9 * answers[f"trap_{m}"]["yes"])
        * (0.5 + answers[f"food_{m}"]["yes"])
        for m in MOVES
    }
    total = sum(scores.values()) or 1.0
    return {m: s / total for m, s in scores.items()}


# --- players ------------------------------------------------------------------


def greedy(game: Game) -> str:
    """The planner acting on its own notes: safe, not a trap, then toward the food."""
    safe = [m for m in MOVES if game.safe(m)] or list(MOVES)
    roomy = [m for m in safe if game.room(m) >= len(game.body)] or safe
    return min(roomy, key=lambda m: (game.distance(game.target(m)), -game.room(m)))


def random_safe(game: Game, rng: random.Random) -> str:
    safe = [m for m in MOVES if game.safe(m)]
    return rng.choice(safe or list(MOVES))


# --- the loop -------------------------------------------------------------------


def bars(distribution: dict[str, float]) -> str:
    return "  ".join(f"{m[0].upper()} {p:5.1%}" for m, p in distribution.items())


PAINT = {"#": "\033[2m#\033[0m", "H": "\033[1;92m@\033[0m", "o": "\033[32mo\033[0m",
         "*": "\033[1;91m*\033[0m", ".": "\033[2m.\033[0m"}


def frame(game: Game, header: str, prefs: dict[str, float] | None, move: str,
          footer: list[str]) -> str:
    """One screen: the board as the model saw it, beside what it made of each move."""
    board = ["".join(PAINT[ch] + " " for ch in row) for row in game.grid().splitlines()]
    panel = [header, f"score {game.score}   length {len(game.body)}", ""]
    for m in MOVES:
        p = prefs[m] if prefs else float(m == move)
        mark = "\033[1m<- played\033[0m" if m == move else ""
        risk = "" if game.safe(m) else "\033[91mfatal\033[0m "
        panel.append(f"{m:<5} {'#' * round(p * 20):<20} {p:6.1%}  {risk}{mark}")
    panel += [""] + footer
    width = 2 * (game.width + 2)  # each cell is drawn as two columns
    rows = max(len(board), len(panel))
    board += [" " * width] * (rows - len(board))
    panel += [""] * (rows - len(panel))
    return "\n".join(f"{b}   {q}" for b, q in zip(board, panel))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="a bigger model if you have the VRAM")
    parser.add_argument("--4bit", dest="quantize", action="store_const", const="4bit",
                        help="load the model in 4 bits (NF4)")
    parser.add_argument("--player", choices=["jeff", "greedy", "random"], default="jeff")
    parser.add_argument("--ask", choices=["each", "menu"], default="each",
                        help="yes/no questions per direction (default), or one four-way choice")
    parser.add_argument("--blind", action="store_true", help="grid only, no per-move notes")
    parser.add_argument("--no-shield", action="store_true", help="play the model's choice even if fatal")
    parser.add_argument("--permutations", default="all", help="N, or 'all' (default)")
    parser.add_argument("--size", type=int, default=8, help="board side, in cells")
    parser.add_argument("--moves", type=int, default=200, help="stop after this many moves")
    parser.add_argument("--max-batch", type=int, default=16,
                        help="rows per forward; lower it if a big model spills out of VRAM")
    parser.add_argument("--quiet", action="store_true", help="one line per game, not per move")
    parser.add_argument("--games", type=int, default=1, help="play this many seeds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="draw the board every move")
    parser.add_argument("--watch", action="store_true",
                        help="live view: redraw the board in place with the model's bars")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to pause per move")
    args = parser.parse_args()

    jeff = None
    if args.player == "jeff":
        jeff = Jeff(*([args.model] if args.model else []), quantize=args.quantize,
                    max_batch=args.max_batch)
    permutations = "all" if args.permutations == "all" else int(args.permutations)

    delay = args.delay if args.delay or not args.watch else 0.15
    if args.watch:
        print("\033[?25l\033[2J", end="")  # hide the cursor, clear once
    try:
        play(args, jeff, permutations, delay)
    finally:
        if args.watch:
            print("\033[?25h", end="")


def play(args, jeff, permutations, delay) -> None:
    totals = {"score": 0, "moves": 0, "overrides": 0, "fatal_wanted": 0, "agree": 0,
              "unstable": 0, "seconds": 0.0}
    for g in range(args.games):
        game = Game(args.size, args.size, seed=args.seed + g)
        rng = random.Random(args.seed + g)
        moves = 0
        print(f"\n\033[1mgame {g + 1}\033[0m  seed {args.seed + g}  player {args.player}"
              + (f"  ask {args.ask}" if jeff else "")
              + ("  blind" if args.blind else "") + ("  unshielded" if args.no_shield else ""))
        while game.alive and moves < args.moves:
            planned = greedy(game)
            note, prefs, stability = "", None, 1.0
            if jeff is not None:
                started = time.perf_counter()
                questions = (menu if args.ask == "menu" else each)(game, args.blind)
                # In menu mode the notes ride on the options, not in the state.
                shared = state(game, blind=args.blind or args.ask == "menu")
                answers = {
                    a.question.key: a
                    for a in jeff.ask(shared, questions, permutations=permutations)
                }
                totals["seconds"] += time.perf_counter() - started
                prefs = preference(answers)
                wanted = max(prefs, key=prefs.__getitem__)
                move = wanted
                if not game.safe(wanted):
                    totals["fatal_wanted"] += 1
                    safe = [m for m in MOVES if game.safe(m)]
                    if safe and not args.no_shield:
                        move = max(safe, key=prefs.__getitem__)
                        totals["overrides"] += 1
                        note = f"  shield: {wanted} was fatal"
                totals["agree"] += wanted == planned
                # The weakest link: one flaky answer is enough to doubt the move.
                stability = min(a.stability for a in answers.values())
                totals["unstable"] += stability < 0.6
                detail = f"{bars(prefs)}  stable {stability:.0%}"
            else:
                move = planned if args.player == "greedy" else random_safe(game, rng)
                detail = ""

            if args.watch:
                header = (f"\033[1mgame {g + 1}\033[0m  move {moves + 1}  "
                          f"{args.model or 'default model' if jeff else args.player}")
                footer = [f"planner would play {planned}"]
                if jeff is not None:
                    footer += [f"stable {stability:.0%} under relabelling",
                               f"{totals['seconds'] * 1000 / (moves + 1):.0f} ms per move"]
                footer.append(f"\033[93m{note.strip()}\033[0m" if note else "")
                print("\033[H" + frame(game, header, prefs, move, footer) + "\033[J", flush=True)
            game.step(move)
            moves += 1
            if args.render:
                print(f"\n{game.grid()}")
            if args.render or (jeff is not None and not args.quiet and not args.watch):
                print(f"  {moves:>3} {move:<5} score {game.score:<3} {detail}{note}")
            if delay:
                time.sleep(delay)

        end = "board full" if game.food is None else ("crashed" if not game.alive else "move limit")
        if args.watch:
            print("\033[H" + frame(game, f"\033[1mgame {g + 1}: {end}\033[0m", None, "", []) + "\033[J")
        print(f"  -> score {game.score}, length {len(game.body)}, {moves} moves, {end}")
        if args.watch and g + 1 < args.games:
            time.sleep(1.5)
        totals["score"] += game.score
        totals["moves"] += moves

    n = max(totals["moves"], 1)
    print(f"\n{args.games} game(s): {totals['score']} food in {totals['moves']} moves")
    if jeff is not None:
        print(
            f"model's own pick was fatal {totals['fatal_wanted']} times "
            f"({totals['fatal_wanted'] / n:.1%}), shield overrode {totals['overrides']}\n"
            f"agreed with the scripted planner on {totals['agree'] / n:.1%} of moves, "
            f"unstable under relabelling on {totals['unstable'] / n:.1%}\n"
            f"{totals['seconds'] * 1000 / n:.0f} ms per move, 0 tokens generated"
        )


if __name__ == "__main__":
    sys.exit(main())
