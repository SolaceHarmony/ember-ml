# AI Agent Guidelines for Ember ML

## Project Ownership & Attribution

**Project Creator & Maintainer:** Sydney Renee (sydney@solace.ofharmony.ai)
**Organization:** The Solace Project

## Git Authorship Policy

### Required Author Attribution

All commits to this repository **MUST** be attributed to:
```
Sydney Renee <sydney@solace.ofharmony.ai>
```

### For AI Agents (Claude Code, Copilot, etc.)

When AI agents interact with this repository:

1. **NEVER modify git user configuration** (global or local)
2. **NEVER add AI co-authorship** unless explicitly requested
3. **ALWAYS verify** authorship before committing:
   ```bash
   git log -1 --format='%an <%ae>'
   ```
4. **Commits made by agents** will automatically inherit the configured human author

### Verification Checklist

Before any git push, verify:
- [ ] `git config user.name` returns "Sydney Renee"
- [ ] `git config user.email` returns "sydney@solace.ofharmony.ai"
- [ ] Recent commits show correct authorship: `git log -5 --format='%an <%ae>'`

## Code Standards for AI Agents

### Python Interpreter

**IMPORTANT:** Always use `python`, not `python3`
- The conda environment is always loaded
- `python3` points to macOS system Python 3.9.6
- Shebang line: `#!/usr/bin/env python`

### Code Integrity

From CLAUDE.md context:
- **NO placeholders** - breaking code is unacceptable
- **NO mockups** - all code must be functional
- **NO simplification** that breaks functionality
- Test theories in new sandboxes, not in stable code

## Development Workflow

### When Making Changes

1. **Read existing code** before modifying
2. **Preserve functionality** - no breaking changes without explicit approval
3. **Test in isolation** if unsure about changes
4. **Commit with descriptive messages** that reflect the actual work done

### Backend & Architecture

- Multi-backend support: NumPy, PyTorch, MLX
- Async operations via `ember_ml.asyncml` (requires Ray)
- Lazy imports for optional dependencies
- Proxy pattern for backend dispatch

---

**Remember:** This is Sydney's project. AI agents are tools to assist, not contributors.
