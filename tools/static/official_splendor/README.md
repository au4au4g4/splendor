# Local official-style Splendor frontend

This directory vendors the repo-owned, local official-style Splendor frontend used by
`tools/splendor_hint_gui.py` at `/official-local-ui`.

It intentionally replaces the hosted page's browser-owned state with same-origin API calls:

- `GET /api/state` is the only game-state source.
- Card, reserved-card, and token controls select actions from `board.legal_moves`.
- Moves are sent through `POST /api/move`.
- Undo is sent through `POST /api/undo`.
- AlphaZero badges and the side panel are rendered from `board.hints` / top-level `hints`.

The hosted `https://cestpasphoto.github.io/splendor.html` iframe mode remains separate and
unchanged because it cannot reliably share internal browser state with the local Python backend.
