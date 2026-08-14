# Space Deck Builder

This is a simulator for a space-based deck building card game. The game has the same exact rules and cards as Star Realms.

## Project Details

- This project uses the uv package manager.
- `torch` is declared **only** in optional-dependency extras, so a bare
  `uv sync` uninstalls it. Always use `uv sync --extra rocm` (or `cuda` / `cpu`).
- Defaults for training, model, and runtime live in `src/config.py` — treat
  that file as the single source of truth rather than duplicating values.
- Card details are available in the data\cards.csv file with the following columns:
  - Set,Qty,Name,Text,Type,Faction,Cost,Defense,Role,Notes

## Important Instructions

- When making comments, ensure they are descriptive and do not just describe the most recent change. Instead, they should describe the final state of the code.