---
title: LayaChess
emoji: ♟️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
short_description: Play chess against Laya, a fine-tuned decision model
---

# LayaChess

Play chess against [Laya](https://huggingface.co/convaiinnovations/laya), a 421M-parameter System 1 decision model,
fine-tuned on 2 million Stockfish-rated moves and wrapped in a Monte Carlo tree search.

- Write-up: https://devroopsaha744.github.io/portfolio/blog/laya-chess/
- Code: https://github.com/devroopsaha744/LayaChess
- Demo video: https://www.youtube.com/watch?v=bPpAlWArs7E
- Model: https://huggingface.co/datafreak/laya-chess
- Run it locally: see the [setup steps](https://github.com/devroopsaha744/LayaChess#run-it-on-your-own-machine)

Laya thinks on ZeroGPU, a shared free GPU, so the first move can take a while and you may wait in a queue.
Laya can think up to 30 seconds a move; that time comes out of each visitor's daily ZeroGPU quota. Stockfish's preferred move is shown next to Laya's for comparison only; it
never picks Laya's moves.
