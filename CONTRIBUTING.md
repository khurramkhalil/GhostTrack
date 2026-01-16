# Contributing to GhostTrack

We welcome contributions to GhostTrack! This document provides guidelines for contributing.

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

File an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)

### Suggesting Features

Open an issue describing:
- The problem your feature would solve
- Your proposed solution
- Alternative solutions considered

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation
7. Commit with clear messages (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints for function signatures
- Add docstrings to all public functions/classes
- Maximum line length: 100 characters
- Use meaningful variable names

### Testing

- Write tests for all new functionality
- Maintain >90% code coverage
- Run full test suite before submitting PR
- Add integration tests for major features

### Documentation

- Update README.md if adding user-facing features
- Add docstrings following Google style
- Update CHANGELOG.md
- Include code examples where appropriate

## Development Setup

```bash
# Clone repository
git clone https://github.com/anthropics/ghosttrack.git
cd ghosttrack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Commit Message Guidelines

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

Example:
```
feat(tracking): Add greedy association option

Implement faster greedy matching as alternative to Hungarian algorithm
for real-time applications. Greedy matching runs in O(n² log n) vs O(n³).

Closes #123
```

## Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create git tag (`git tag -a v1.0.0 -m "Version 1.0.0"`)
4. Push tag (`git push origin v1.0.0`)
5. GitHub Actions will create release

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing to GhostTrack!
