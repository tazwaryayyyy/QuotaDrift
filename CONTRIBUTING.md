# Contributing to QuotaDrift

Thank you for your interest in contributing to QuotaDrift! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for testing)
- API keys for at least one AI provider

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/tazwaryayyyy/quotadrift.git
   cd quotadrift
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the Application**
   ```bash
   python main.py
   ```

6. **Verify Setup**
   - Visit http://localhost:8000
   - Check that the application loads correctly
   - Test the `/api/health` endpoint

## 🧪 Running Tests

### Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=quotadrift --cov-report=html

# Run specific test file
pytest tests/test_router.py -v

# Run with markers
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
```

### Test Structure

```
tests/
├── test_router.py          # Router and provider tests
├── test_memory.py          # Memory and storage tests
├── test_config.py          # Configuration tests
├── test_api.py             # API endpoint tests
└── conftest.py             # Test fixtures and utilities
```

### Writing Tests

- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies (API calls, databases)
- Include both unit and integration tests
- Test error conditions and edge cases

Example:
```python
def test_provider_failover():
    """Test that the router fails over to the next provider."""
    # Arrange
    mock_router = create_mock_router()
    
    # Act
    response = mock_router.chat(messages=[{"role": "user", "content": "test"}])
    
    # Assert
    assert response["content"] is not None
    assert response["model_used"] is not None
```

## 🎨 Code Style

### Linting and Formatting

We use `ruff` for linting and `black` for formatting.

```bash
# Check code style
ruff check .

# Format code
ruff format .
black .

# Check formatting without modifying
black --check .
```

### Code Style Guidelines

- Follow PEP 8
- Use type hints for all functions
- Keep lines under 88 characters
- Use descriptive variable and function names
- Add docstrings for all public functions and classes

### Example Code Style

```python
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

def process_message(
    message: str, 
    session_id: int, 
    model: str = "default"
) -> Dict[str, str]:
    """
    Process a user message and return a response.
    
    Args:
        message: The user's input message
        session_id: The session identifier
        model: The AI model to use
        
    Returns:
        Dictionary containing the response and metadata
        
    Raises:
        ValueError: If message is empty or invalid
    """
    if not message.strip():
        raise ValueError("Message cannot be empty")
    
    logger.info(f"Processing message for session {session_id}")
    
    return {
        "response": "Processed response",
        "model_used": model,
        "session_id": session_id
    }
```

## 📝 Submitting Changes

### Commit Guidelines

- Use clear, descriptive commit messages
- Start with a verb (Fix, Add, Update, etc.)
- Keep commits focused and atomic
- Reference issues when applicable

Examples:
```
Add: Provider health check endpoint
Fix: Silicon Flow API base URL configuration
Update: README with new provider information
Fixes: #123 - Handle empty message validation
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run Quality Checks**
   ```bash
   pytest tests/ && ruff check . && black .
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: Your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use descriptive title and description
   - Link to relevant issues
   - Include screenshots for UI changes
   - Describe testing performed

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Reporting Issues

### Bug Reports

Use the provided bug report template in `.github/ISSUE_TEMPLATE/bug_report.md`.

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs or screenshots

### Feature Requests

Use the provided feature request template in `.github/ISSUE_TEMPLATE/feature_request.md`.

Include:
- Clear description of the feature
- Use case and motivation
- Implementation ideas (optional)
- Potential alternatives considered

## 🔧 Development Tools

### Recommended IDE Setup

- **VS Code**: Install Python, Pylance, and Docker extensions
- **PyCharm**: Configure Python interpreter and code style
- **Git**: Use pre-commit hooks for quality checks

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Debugging

- Use Python's built-in debugger (`pdb`)
- Add logging statements for troubleshooting
- Use VS Code's debugger for breakpoints
- Test with different provider configurations

## 📚 Documentation

### Updating Documentation

- README.md: Project overview and setup
- API docs: Update docstrings for new endpoints
- Comments: Add inline comments for complex logic
- Examples: Provide usage examples for new features

### Documentation Style

- Use clear, concise language
- Include code examples
- Use proper formatting and structure
- Keep documentation up-to-date with code changes

## 🤝 Community Guidelines

### Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discord**: Join our community Discord (link in README)
- **Documentation**: Check existing docs and issues first

### Communication

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## 🏆 Recognition

### Contributors

All contributors are recognized in:
- README.md contributors section
- GitHub contributors list
- Release notes for significant contributions

### Types of Contributions

- Code contributions (features, bug fixes)
- Documentation and examples
- Bug reports and feedback
- Community support and discussions
- Design and UX improvements

## 📋 Release Process

### Versioning

We follow semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Breaking changes: Increment MAJOR
- New features: Increment MINOR
- Bug fixes: Increment PATCH

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Tag created
- [ ] Release notes published

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Additional AI provider integrations
- Enhanced error handling and logging
- Performance optimizations
- Security improvements

### Medium Priority
- UI/UX improvements
- Additional test coverage
- Documentation enhancements
- Docker and deployment improvements

### Low Priority
- Code refactoring
- Minor feature enhancements
- Tooling and automation improvements

## 📞 Contact

- **Maintainers**: [Tazwar Yayyyy](https://github.com/tazwaryayyyy)
- **Email**: Use GitHub Discussions for public questions
- **Issues**: [GitHub Issues](https://github.com/tazwaryayyyy/quotadrift/issues)
- **Security**: Report security issues privately

---

Thank you for contributing to QuotaDrift! Your contributions help make this project better for everyone. 🚀
