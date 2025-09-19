# Contributing to VAE Project

We welcome contributions to this Variational Autoencoder project! This document provides guidelines for contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/vae-project.git
   cd vae-project
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Development Setup

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep line length under 100 characters

### Testing
- Write tests for new features
- Ensure existing tests pass
- Test with multiple datasets (MNIST, Fashion-MNIST, CIFAR-10)

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Update docstrings when modifying functions

## 📝 How to Contribute

### Reporting Bugs
1. Check if the bug has already been reported in Issues
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Error messages and stack traces

### Suggesting Features
1. Check existing issues for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Possible implementation approach

### Code Contributions

#### Branch Naming
- `feature/description` for new features
- `bugfix/description` for bug fixes
- `docs/description` for documentation updates
- `refactor/description` for code refactoring

#### Commit Messages
Follow conventional commit format:
- `feat: add new visualization feature`
- `fix: resolve training instability issue`
- `docs: update installation instructions`
- `refactor: improve model architecture`

#### Pull Request Process
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   python train.py --dataset mnist --epochs 5  # Quick test
   streamlit run app.py  # Test UI
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Create a Pull Request** with:
   - Clear title and description
   - Reference related issues
   - Screenshots for UI changes
   - Test results

## 🎯 Areas for Contribution

### High Priority
- [ ] Add more dataset support (CelebA, SVHN, etc.)
- [ ] Implement additional VAE variants (WAE, InfoVAE, etc.)
- [ ] Add comprehensive unit tests
- [ ] Improve documentation with tutorials

### Medium Priority
- [ ] Add data augmentation options
- [ ] Implement model comparison features
- [ ] Add export functionality for trained models
- [ ] Create Jupyter notebook tutorials

### Low Priority
- [ ] Add more visualization options
- [ ] Implement custom loss functions
- [ ] Add model interpretability features
- [ ] Create Docker containerization

## 🧪 Testing Guidelines

### Manual Testing Checklist
- [ ] Training completes without errors
- [ ] Streamlit app loads and functions properly
- [ ] Visualizations generate correctly
- [ ] Model saves and loads properly
- [ ] All datasets work correctly

### Automated Testing
We encourage adding automated tests for:
- Model architecture validation
- Data loading and preprocessing
- Training loop functionality
- Visualization generation

## 📋 Code Review Process

All contributions go through code review:

1. **Automated checks** must pass (when implemented)
2. **Manual review** by maintainers
3. **Testing** on different environments
4. **Documentation** review for clarity

### Review Criteria
- Code quality and style
- Functionality and correctness
- Performance impact
- Documentation completeness
- Test coverage

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributor graphs

## 📞 Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers for sensitive issues

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement
Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct.

## 🙏 Thank You

Thank you for contributing to this project! Your efforts help make this a better resource for the machine learning community.

---

**Happy Contributing! 🎉**
