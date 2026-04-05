# Variant 9 Migration Notes

## What changed

- Dropped direct `litellm` dependency (comes transitively via openhands-sdk)
- Widened openhands version bounds from `<1.16.1` to `<2`
- Added CI workflow, publish workflow, codecov config
- Added hatch environment configs for testing, types, coverage
- Pytest now skips LLM tests by default (`-m 'not llm'`)
- Removed `.env.example`, `.python-version`, `assets/shallnotpass.png`

## Breaking changes

- The widened openhands bounds (`<2`) may pull a newer openhands-sdk version that could have different behavior. The `<1.16.1` pin was specifically added to work around `cannot import name 'DeclaredResources' from 'openhands.sdk.tool'`. If the newer version reintroduces this, you'll need to re-pin.
- Removing `litellm` as a direct dep means it's no longer version-locked. The version resolved will be whatever openhands-sdk brings.

## Migration

If you pinned specific litellm or openhands versions in your packaging, verify compatibility. Run `pip install -e .` and check `pip show litellm openhands-sdk` to see resolved versions.
