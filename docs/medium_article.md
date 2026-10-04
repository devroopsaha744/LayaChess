# I Taught a Decision Model to Play Chess Over a Weekend (and It Only Cried a Little)

*How I took a 421M-parameter AI that had never seen a chessboard, fed it 2 million Stockfish opinions on free GPUs, and built a chess engine around it. With crashes, cooked laptops and one very humbling Stockfish.*

![LayaChess learning curve](learning_curve.png)

---

## The idea (a.k.a. "how hard could it be?")

It was a Thursday. I was at work. I had half a day plus a weekend, and I had just read about **decision models**: AI models that don't write essays, they **make choices**. You hand them a situation and a question, and they score the options. Clean. Fast. No rambling.

Two of them caught my eye:

- **Jev** by TypeSafe AI: closed API, strong out of the box, and I had exactly zero API keys. Moving on.
- **Laya** by Convai Innovations: open source (Apache 2.0), a 421M-parameter ModernBERT-large under the hood, and it comes with a fine-tuning notebook that runs on Kaggle's free T4 GPUs. 👀

And chess is *literally* a decision problem. Every turn: here's the board, here are your legal moves, pick one. No illegal moves possible, because you only ever offer legal ones. (Looking at you, chatbots that castle through their own pieces.)

So the plan wrote itself:

> Teach Laya chess. Compare it with Stockfish and Leela. Do it in a weekend. On free GPUs.

Reader, it was not that simple. But it worked.

---

## Step 0: What does an untrained Laya think of chess?

Before training, I asked base Laya to pick an opening move as White.

It said **b4**. (The Polish Opening. Bold. Respectable among the hipsters.)

Then as Black, against 1.e4, it went **h6**. 🫠

And when I set up the classic Scholar's Mate, where White can play Qxf7# and win on the spot, Laya preferred… **d3**. A quiet pawn move. While checkmate sat right there.

Best-move accuracy against Stockfish on held-out positions: **6%**. Picking a random legal move gets you about 3%. So: twice as good as a coin flip with extra steps.

We had work to do.

---

## Step 1: The data, a.k.a. "please don't make me run Stockfish all night"

My original plan was to take Lichess games, run Stockfish on a million positions overnight, and label every move. My laptop was not excited about this plan.

Then I remembered DeepMind already did it, at a slightly larger scale. Their **ChessBench** dataset (from the "searchless chess" paper) has **15.3 billion** positions-and-moves, each labeled with Stockfish's win probability. Every legal move. Already scored.

Small problem: it's stored in a custom `.bag` format, normally read with Apache Beam. I didn't want Apache Beam in my life, so I wrote a tiny reader instead. A `.bag` file is just records back to back, with an index of byte offsets at the end, and each record is `(position, move, win chance)`. About 30 lines of Python, no Beam required.

I didn't need 15 billion examples (that would take my free GPUs roughly 120 years; I did the math and then lay down for a bit). I grabbed **the test set + 2 training files**, about 3.2 GB, and saved them as a Kaggle dataset. Kaggle even downloaded them for me, so not a single byte went through my laptop.

---

## Step 2: How do you ask a decision model about chess?

Laya speaks in questions with options. So for every legal move, I ask:

> *"black plays Rg4 (rook g6-g4). Win chance for black?"*
> Options: 0–10%, 10–20%, … 90–100%

The board goes in as context:

```json
{"to_move": "black",
 "white": "Kg1 Qc5 Rd1 Rf1 Bc4 Pf2 Pg2 Ph2 Pe3 Pb4 Pa5",
 "black": "Kg8 Qe5 Rg6 Re8 Nd5 Pa6 Pc6 Ph6 Pb7 Pf7 Pg7",
 "castling": "-", "en_passant": "-"}
```

The model answers with a probability over the 10 levels, I take the average, and that's the move's win chance. To play, I score every legal move and pick the highest one.

This is the same recipe DeepMind found works best ("action-value prediction"): judge each move by how winning the position is *after* it. The twist is that my model is a general decision model answering in its own question format, not a transformer built for chess.

Fun fact: my **first** version described the board as an 8×8 grid and used 16 win-chance levels. That came to **335 tokens per question** and trained at a snail-like **49,000 examples per hour**. Switching to piece lists and 10 levels cut it to **~194 tokens** and **doubled the speed to ~97,000 per hour**. Same idea, half the words. Turns out models, like people, prefer shorter questions.

---

## Step 3: Training on free GPUs (the crash compilation)

Kaggle gives you 2× NVIDIA T4s, about 30 GPU hours a week, and runs of up to 12 hours. Here's how it went:

**Crash #1: "StopIteration in replica 1."**
I split each batch across the two GPUs with PyTorch's `DataParallel`. ModernBERT, the model inside Laya, asked "what device am I on?" by looking at its own parameters… and in the GPU copies, it had none. So it panicked. Fix: a tiny patch telling it where to look.

**Crash #2: Out of memory.**
421M parameters + the optimizer's memory = a T4 begging for mercy. Fix: the notebook now *tests* memory before training and only switches on the slower tricks (gradient checkpointing, smaller batches) when it actually needs them.

**Crash #3: Me.**
I started the second training run before the first one finished. It grabbed a half-finished checkpoint, and both runs tried to upload to the same Hugging Face repo. Fix: cancel, breathe, wait for the first run to finish, try again.

Along the way I also learned that tokens you paste into a chat should be deleted afterwards, and that a deleted token in a Kaggle secret produces a very confident "Invalid username or password." 🙃

Once it was stable, each run trained for 11 hours, saved a checkpoint every 45 minutes, and uploaded it to Hugging Face. Run 2 picked up exactly where run 1 stopped: same step, same position in the data, same optimizer state.

---

## Step 4: Did it actually learn?

Yes! 🎉

| Training examples | Best-move accuracy | Win-chance error |
|---|---|---|
| 0 (base Laya) | 6% | 28.6 points |
| 128k | 16% | 12.9 |
| 512k | 22% | 10.4 |
| 1.03M (end of run 1) | 24% | 9.1 |
| **2.05M (end of run 2)** | **27%** | **8.2** |

*(Best-move accuracy = how often its #1 move matches Stockfish's #1 move, on 300 positions from games it never saw. Win-chance error = how far off its win estimate is, on average.)*

A few things I learned from staring at these numbers at 2 a.m.:

- **The first 100k examples do the heavy lifting.** The model quickly figures out *"how good is this position?"* Then it spends the next million learning the much harder *"which move is better?"*
- **The best-move number is noisy.** With 300 test positions it wobbles about ±2.5 points. At one point it dropped from 21.7% to 19.7% and I briefly considered a career change. The loss kept improving, so it was just noise.
- **Restarting the learning rate causes a dip.** At the start of run 2, the scores got slightly worse for a couple of hours, then recovered and beat run 1. Expected, but still stressful.

And the vibe check:

- Opening move: **d4** ✅ (not b4)
- Reply to 1.e4: **d6, d5, e6, Nf6, c6**: all real openings ✅ (not h6)
- Scholar's Mate position: **Qxf7# with 94% confidence**, next best at 33% ✅
- A free queen sitting there: **takes it, 94%** ✅

From "b4 and h6" to "takes your queen and mates you in one" over a weekend. I'm unreasonably proud.

---

## Step 5: Building an actual engine

A model that scores moves isn't an engine yet. So I built the rest:

- **Search: MCTS (Monte Carlo tree search)**, the same family of algorithm Leela Chess Zero and AlphaZero use. Laya's move scores decide which moves to explore first, and its win chances tell the search how good a position is. Checkmates and draws come straight from the rules, never from the model.
- **UCI support**, the standard engine protocol, so it plugs into chess apps, match tools and Lichess bots.
- **A browser board**: play against it locally, and after every move it shows its win chance for its top candidate moves. It's weirdly satisfying to watch it "think".
- **A match runner** against Stockfish, with openings, both colours, PGN files and an Elo estimate.

I tested the search with a fake "material counting" model first: without search it happily grabbed a pawn defended by another pawn and lost its queen; with just 20 positions of search it saw the trap. Search works. 🧠

---

## Step 6: The humbling (Stockfish enters the chat)

Time to find out how strong it is.

On my MacBook, without search, LayaChess played Stockfish at **UCI_Elo 1320**, Stockfish's weakest official setting.

It lost. Then it lost again. The third game lasted **88 moves** (a real fight!), and then it lost.

Meanwhile my laptop was running its GPU at 100% and getting hot enough to make toast.

So I moved the matches to Kaggle *with* search… and it lost the first three there too. 😭

Then I went down a rabbit hole and found out something fun: you **can't make Stockfish weaker than 1320**. In Stockfish's source code, Elo 1320 maps to "Skill Level 0", the lowest it goes. Most people think of "1320" as beginner level, but Stockfish 1320 doesn't hang pieces the way beginners do. It's a sneakily solid opponent.

The honest summary:

> LayaChess v2 plays real chess (sensible openings, finds mates, punishes free pieces), but it's still weaker than Stockfish at its lowest setting.

And honestly? That makes sense:

- **DeepMind trained on 15 billion examples. I used 2 million.** That's 7,500× less data.
- **Search is slow.** Laya reads the whole board once *per legal move*. That's about 35 passes of a 421M model per position: about 2.5 positions per second on a T4, versus millions for Stockfish.

---

## What's next: v3 (same brain, faster answers)

The fix for the speed problem is to stop asking one question per move.

**v3 reads the board once** with the same Laya encoder I trained this weekend, and a small new "move head" scores **every legal move in one pass**. That's the design Leela Chess Zero uses, with Laya's brain inside.

- About **35× faster** per position, so search can look at hundreds of positions per move instead of about 30.
- About **2× shorter input** (94 tokens instead of 194), so training is faster too.
- It **starts from v2's trained encoder**, so this weekend's training isn't thrown away. It's the foundation.

Then: a **Lichess bot**, so it gets a real rating from real games, and so you can challenge it yourself.

---

## Things I'd tell myself on Thursday

1. **Read the dataset docs before planning to label your own.** Someone may already have done it with 15 billion examples.
2. **Shorter inputs = faster training.** Half the tokens, double the speed.
3. **Never judge a training run by one number.** Watch the loss and the error; the accuracy will catch up.
4. **Save checkpoints somewhere outside the machine.** Kaggle runs end, laptops overheat, and I start things too early.
5. **"Elo 1320" is not a beginner.** Respect the fish. 🐟
6. **A general-purpose model *can* learn chess.** It just needs a lot more practice than one weekend allows.

---

## Links

- 🤗 Model: huggingface.co/datafreak/laya-chess
- 💻 Code (training notebooks, engine, browser board): github.com/devroopsaha744/LayaChess
- ♟️ Laya by Convai Innovations · ChessBench by Google DeepMind · Stockfish · python-chess

*If you've got a spare GPU and a chess grudge, come play it. It will take your free queen.*

---

<!-- Notes for publishing (delete before posting):
- Medium: import this file or paste section by section; upload learning_curve.png where it's referenced.
- Add a GIF or screenshot of the browser board right after "Step 5" (python -m laya_chess.play).
- If the repos stay private, remove the two links or replace them with "coming soon".
-->
