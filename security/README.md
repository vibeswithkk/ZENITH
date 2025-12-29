# Zenith Security

Security policy, audit tools, and vulnerability reporting.

## Security Scanning Tools

### 1. Security Audit Runner

Comprehensive security scan including SAST and dependency checks.

**Prerequisites:**
```bash
pip install bandit safety pip-audit
```

**Usage:**
```bash
# Full security audit
python security/audit.py

# SAST only
python security/audit.py --sast-only

# Dependency scan only
python security/audit.py --deps-only

# Export JSON report
python security/audit.py --output report.json
```

---

### 2. Fuzzing Tests

Property-based testing with Hypothesis.

**Prerequisites:**
```bash
pip install hypothesis pytest
```

**Usage:**
```bash
# Run all fuzz tests
pytest security/fuzz_tests.py -v

# With statistics
pytest security/fuzz_tests.py --hypothesis-show-statistics
```

---

## CI/CD Integration

Security scans run automatically on:
- Every push to `main` and `develop`
- Every pull request
- Weekly scheduled scan (Monday 00:00 UTC)

See `.github/workflows/security.yml`

---

## Vulnerability Reporting

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email: security@zenith-project.org (or create private advisory)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to respond within 48 hours.

---

## Security Checklist

| Check | Tool | Frequency |
|-------|------|-----------|
| SAST | Bandit | Every commit |
| Dependency CVEs | Safety, pip-audit | Every commit |
| Fuzzing | Hypothesis | Every commit |
| Secrets scan | Custom patterns | Every commit |

---

## Attack Vectors Mitigated

1. **Injection**: Input validation fuzzing
2. **Dependency Hijacking**: CVE scanning
3. **Hardcoded Secrets**: Pattern detection
4. **Unsafe Deserialization**: pickle usage detection
5. **Command Injection**: shell=True detection
