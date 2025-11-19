# Contributing to Schedule Engine

Thank you for your interest in contributing to the Schedule Engine project!

##  Development Setup

### Prerequisites
- Python 3.11+
- UV package manager
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/krishna-ji/schedule-engine.git
cd schedule-engine

# Install UV and setup environment
python setup-uv
```

##  Code Standards

### Python Style
- **PEP 8 compliant** - Follow Python style guidelines
- **Type hints** for function parameters and returns
- **Docstrings** for all modules, classes, and functions
- **Line length**: 88 characters (Black default)

### Import Order
```python
# 1. Standard library
import sys
from pathlib import Path

# 2. Third-party packages
import numpy as np
from deap import tools

# 3. Local modules
from src.config import get_config
from src.core.types import Individual
```

##  Testing

### Run Tests
```bash
# All tests
pytest test/

# With coverage
pytest --cov=src --cov-report=html test/

# Specific test
pytest test/unit/test_config_loader.py
```

##  Commit Guidelines

### Format
```
<type>(<scope>): <summary>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Test additions
- `doc`: Documentation

### Examples
```bash
feat(rl): add specialist agents
fix(constraints): correct hypervolume calculation
doc(readme): add runtime modes reference
```

##  Pull Request Process

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Make changes and test
4. Commit with descriptive messages
5. Push to fork
6. Open Pull Request

##  Questions?

- Check existing issues on GitHub
- Review documentation in `docs/`
- Contact maintainers: Krishna Acharya, Dinanath Padhya, Bipul Dahal

---

**Thank you for contributing!** 
