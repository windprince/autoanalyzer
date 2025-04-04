# AutoAnalyzer Project Guidelines

## Commands
- Run app: `python app.py`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/test_file.py::test_function`
- Code check: `flake8 . && mypy .`

## Code Style

### Python
- Use type hints (see DataProfiler.py)
- Use docstrings with """triple quotes"""
- Classes: PascalCase; Methods/functions: snake_case
- Private methods prefixed with underscore (_method_name)
- Error handling with try/except (see DataProfiler._is_datetime)
- Return type annotations for function definitions

### JavaScript
- Classes: PascalCase; Methods/variables: camelCase
- Use modern ES6+ syntax with classes and async/await
- Import dependencies at top of file
- Document methods with descriptive comments

### General
- Organize related classes into separate files
- Keep methods focused and single-purpose
- Favor explicit variable names over abbreviations
- Add error handling for user inputs and data processing
- Document all public interfaces