# Quick Reference: Virtual Environment Commands

## Setup (One-time)

```powershell
# Windows PowerShell
.\setup-venv.ps1
```

```bash
# Linux/Mac
./setup-venv.sh
```

## Daily Usage

### Activate Environment

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/Mac
source .venv/bin/activate
```

### Run Schedule Engine

```bash
# After activating the environment
python main.py --env test    # Quick test (10 generations)
python main.py --env dev     # Medium run (100 generations)
python main.py --env prod    # Full quality (200+ generations)
```

### Deactivate Environment

```bash
deactivate
```

## Managing Dependencies

### Install New Package

```bash
pip install package-name
pip freeze > requirements.txt  # Update requirements
```

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### View Installed Packages

```bash
pip list
```

## Notes

-  `.venv/` is already in `.gitignore`
-  No conda required
-  Python 3.8+ compatible (tested with 3.13)
-  Full documentation: `VENV_SETUP.md`
