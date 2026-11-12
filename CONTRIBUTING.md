# Contributing

## Workflow

1. Create a branch: `feat/<topic>`, `fix/<topic>` or `docs/<topic>` (e.g. `feat/mass-properties`).
2. Make the change with a test. Mapping changes need a unit test using a tiny USD fixture in `tests/fixtures/` or built in `tests/conftest.py`.
3. `make lint && make test`.
4. Update `docs/usd_to_mjcf_mapping.md` (status column) and `CHANGELOG.md`.
5. Open a PR. The template has the checklist.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `feat: map MassAPI to <inertial>`, `fix: convert angular drive gains to radians`, `docs: …`, `test: …`, `chore: …`.

## Design decisions

Anything that's hard to reverse (new dependency, data format, architecture change) gets a short record in `docs/decisions/NNNN-title.md`. Copy `0001` as a template.

## Data

Never commit downloaded assets or outputs. `data/` and `outputs/` are git-ignored.
