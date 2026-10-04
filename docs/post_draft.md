# LinkedIn / blog post draft (LayaChess v2)

I taught a general-purpose AI decision model to play chess over a weekend, on free GPUs. ♟️

Laya (by Convai Innovations) is a 421M open-source "decision model": you give it a situation and a question, and it
scores the options. It was never trained on chess. Out of the box it opened with b4 and answered 1.e4 with h6.

So I fine-tuned it on DeepMind's ChessBench data, asking it one question per legal move:
"White plays Nf3. What's White's chance of winning now?"

Results after 2 Kaggle sessions (2.05M training examples, 2× T4):
→ Picks Stockfish's best move 27% of the time, up from 6% (random ≈ 3%)
→ Win-chance error down from 28.6 to 8.2 percentage points
→ Finds mate-in-one with 94% confidence and takes free pieces

Then I built an engine around it: Leela-style tree search (MCTS) guided by Laya's move scores, a UCI interface so it
works with any chess app, and a browser board where you can play it and watch its win chance for every candidate move.

What I learned:
• DeepMind trained on 15 billion examples; I used 2 million. It learns fast, but data is the bottleneck.
• Asking a 421M model one question per move makes search slow (~2.5 positions/s on a T4). Next version reads the
  board once and scores every move in one pass (~35× faster), keeping the same trained brain.
• Honest strength check: without search, it still loses to Stockfish at its weakest setting. A Lichess bot rating is
  coming.

Model: huggingface.co/datafreak/laya-chess
Code: github.com/devroopsaha744/LayaChess

#MachineLearning #Chess #AI #DeepLearning #OpenSource

---
Attach: docs/learning_curve.png + a 30–60 s screen recording of a game on the browser board
(`python -m laya_chess.play`, set "Laya thinks: Instantly" for a snappy video).
Remove the links if the repos stay private.
