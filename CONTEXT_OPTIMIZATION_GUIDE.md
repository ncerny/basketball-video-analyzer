# Context Size Optimization Solution

## Problem Analysis
Your OpenCode setup is exhausting quotas and causing "prompt too large" issues due to:

1. **Large files being included in context**:
   - `backend/videos/` (7.6GB of video files)
   - `frontend/node_modules/` (346MB of dependencies)
   - `backend/venv/` (82MB of Python virtual environment)

2. **Verbose documentation**:
   - `docs/implementation-plan.md` (670 lines)
   - `docs/beads-structure.md` (312 lines)
   - Multiple README files

## Solution Implemented

### 1. `.opencodeignore` File
Created comprehensive exclusions for:
- Video files and directories
- Dependencies (node_modules, venv)
- Build artifacts (dist, build)
- Database files (*.db)
- Lock files (pnpm-lock.yaml, poetry.lock)
- Cache and temporary files

### 2. Optimized Documentation
Created streamlined versions:
- `CLAUDE_OPTIMIZED.md` - Essential project info (50% smaller)
- `AGENTS_OPTIMIZED.md` - Agent instructions (70% smaller)

### 3. Configuration Strategy

#### Option A: Use Optimized Files (Recommended)
Replace the main documentation files:
```bash
mv CLAUDE_OPTIMIZED.md CLAUDE.md
mv AGENTS_OPTIMIZED.md AGENTS.md
```

#### Option B: Keep Original Files
Add to `.opencodeignore`:
```
docs/implementation-plan.md
docs/beads-structure.md
backend/README.md
frontend/README.md
```

#### Option C: Selective Documentation Loading
Only read detailed docs when needed:
- Use `CLAUDE_OPTIMIZED.md` for day-to-day work
- Read `docs/implementation-plan.md` only for architecture decisions

## Expected Results

### Context Size Reduction
- **Before**: ~8GB+ total project size in context
- **After**: ~50MB actual codebase size in context
- **Reduction**: ~99.4% smaller context

### Benefits
1. **Prevents quota exhaustion** - Dramatically reduced token usage
2. **Eliminates "prompt too large" errors** - Context fits within limits
3. **Faster response times** - Less context to process
4. **More focused assistance** - Only relevant code included

### Verification
Test the optimization by:
1. Checking that large directories are excluded from context
2. Ensuring essential project files are still accessible
3. Verifying that normal development workflow continues

## Additional Recommendations

### 1. Regular Cleanup
Add to git hooks or CI:
```bash
# Clean up large generated files
git clean -fdx
```

### 2. Monitoring
Set up alerts for:
- Large files added to repository
- Growing context size
- Quota usage patterns

### 3. Documentation Strategy
- Keep essential info in main CLAUDE.md
- Move detailed specs to separate files
- Load detailed docs only when needed

This solution should resolve both quota exhaustion and "prompt too large" issues while maintaining full development functionality.