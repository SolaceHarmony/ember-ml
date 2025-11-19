# Claude Code Instructions for Ember ML

## Git Commit Authorship

**CRITICAL:** All commits must be attributed to the human author only.

**Author Information:**
- **Name:** Sydney Renee
- **Email:** sydney@solace.ofharmony.ai
- **Organization:** The Solace Project

### Git Configuration Requirements

Before making any commits, verify the git configuration:

```bash
# Global config (already set correctly)
git config --global user.name "Sydney Renee"
git config --global user.email "sydney@solace.ofharmony.ai"

# Check author of recent commits
git log -1 --format='%an <%ae>'
```

### AI Agent Guidelines

When Claude Code or other AI agents make commits:
- **DO NOT** change git authorship settings
- **DO NOT** add AI co-authorship to commits
- All commits will automatically be attributed to Sydney Renee
- The only acceptable co-authorship tag is when explicitly requested by the human author

### Verification

Always verify authorship before pushing:
```bash
git log --format='%an <%ae>' -n 5
```

All commits should show: `Sydney Renee <sydney@solace.ofharmony.ai>`

---

*This ensures proper attribution to the human creator and maintainer of this project.*
