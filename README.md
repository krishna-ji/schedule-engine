# Schedule Engine Project
[BEI-Major Project]
- Krishna Acharya
- Dinanath Padhya
- Bipul Dahal
- Claude काका
- Copilot मामा

## Quick Start

### Installation with UV ⚡ (10-100x faster than pip)

**One-line setup (Windows/Linux/macOS):**
```bash
# Run the setup script (auto-installs UV if needed)
python setup-uv
```

**Manual setup:**
```bash
# 1. Install UV (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment and install dependencies
uv venv .venv
uv sync
```

### Running the Engine

**Production mode (all CPU cores):**
```bash
uv run prod
```

**Test mode (4 workers, memory-safe):**
```bash
uv run test
```

**Custom configuration:**
```bash
python main.py --config configs/cpsat.prod.yaml   # Production
python main.py --config configs/cpsat.test.yaml   # Test
```

### Remote VM Deployment

**Quick setup on VM:**
```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone/upload project
cd orSolver

# 3. Setup environment
python setup-uv

# 4. Run scheduler
uv run prod      # Production (all cores)
# or
uv run test      # Test mode (4 workers)
```

**For long-running processes (detached mode):**
```bash
# Run in background with nohup
nohup uv run prod > solver.log 2>&1 &

# Monitor progress
tail -f solver.log

# Check process status
ps aux | grep python
```

---

## Tmux Guide for Remote Sessions

### Basic Commands
```bash
# Session Management
tmux new -s <name>          # Create new session
tmux ls                     # List all sessions
tmux attach -t <name>       # Attach to session
tmux kill-session -t <name> # Kill session
```

**Detach from session:** `Ctrl + b`, then `d` (keeps process running)

### Window Management
| Action | Command |
|--------|---------|
| New window | `Ctrl + b`, then `c` |
| Next window | `Ctrl + b`, then `n` |
| Previous window | `Ctrl + b`, then `p` |
| Rename window | `Ctrl + b`, then `,` |
| Close window | `exit` or `Ctrl + b`, then `&` |

### Pane Management (Split Screen)
| Action | Command |
|--------|---------|
| Split vertically | `Ctrl + b`, then `%` |
| Split horizontally | `Ctrl + b`, then `"` |
| Switch pane | `Ctrl + b`, then arrow keys |
| Resize pane | `Ctrl + b`, then hold arrow keys |
| Close pane | `exit` |

### Scroll & Copy Mode
| Action | Command |
|--------|---------|
| Enter scroll mode | `Ctrl + b`, then `[` |
| Navigate | Arrow keys or `PgUp`/`PgDn` |
| Scroll fast | `Ctrl + u` (up), `Ctrl + d` (down) |
| Start selection | `Space` |
| Copy selection | `Enter` |
| Paste | `Ctrl + b`, then `]` |
| Exit scroll mode | `q` |

---

## Documentation

See `docs/` folder for detailed documentation and deployment checklist:
- `docs/DEPLOYMENT_CHECKLIST.md` - Production deployment guide
- `docs/QUICK_START.md` - Getting started
- `docs/UV_QUICKSTART.md` - UV package manager guide

