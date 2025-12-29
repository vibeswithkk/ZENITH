#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Security Audit Runner

Comprehensive security scanning including:
- SAST (Static Application Security Testing) via Bandit
- Dependency vulnerability scanning via safety/pip-audit
- Code quality checks

Usage:
    python security/audit.py
    python security/audit.py --sast-only
    python security/audit.py --deps-only
    python security/audit.py --output report.json

Reference:
    OWASP ASVS v4.0, NIST SP 800-53
"""

import argparse
import json
import subprocess
import sys
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SecurityFinding:
    """Individual security finding."""

    severity: str
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class AuditReport:
    """Complete security audit report."""

    timestamp: str
    zenith_version: str
    scan_duration_seconds: float
    sast_findings: list = field(default_factory=list)
    dependency_findings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_sast_finding(self, finding: SecurityFinding) -> None:
        """Add a SAST finding to the report."""
        self.sast_findings.append(asdict(finding))

    def add_dependency_finding(self, finding: SecurityFinding) -> None:
        """Add a dependency finding to the report."""
        self.dependency_findings.append(asdict(finding))

    def compute_summary(self) -> None:
        """Compute summary statistics."""
        sast_by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in self.sast_findings:
            sev = f.get("severity", "LOW").upper()
            if sev in sast_by_severity:
                sast_by_severity[sev] += 1

        self.summary = {
            "total_sast_findings": len(self.sast_findings),
            "total_dependency_findings": len(self.dependency_findings),
            "sast_by_severity": sast_by_severity,
            "passed": (
                sast_by_severity["HIGH"] == 0 and len(self.dependency_findings) == 0
            ),
        }

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(asdict(self), indent=2)

    def print_report(self) -> None:
        """Print formatted report to console."""
        print("\n" + "=" * 60)
        print("  ZENITH SECURITY AUDIT REPORT")
        print("=" * 60)
        print(f"  Timestamp: {self.timestamp}")
        print(f"  Version: {self.zenith_version}")
        print(f"  Duration: {self.scan_duration_seconds:.2f}s")
        print("-" * 60)

        # SAST Summary
        print("\n  SAST FINDINGS:")
        if self.sast_findings:
            for sev in ["HIGH", "MEDIUM", "LOW"]:
                count = self.summary.get("sast_by_severity", {}).get(sev, 0)
                if count > 0:
                    print(f"    {sev}: {count}")
        else:
            print("    No issues found")

        # Dependency Summary
        print("\n  DEPENDENCY VULNERABILITIES:")
        if self.dependency_findings:
            for f in self.dependency_findings[:5]:
                print(f"    - {f.get('description', 'Unknown')}")
            if len(self.dependency_findings) > 5:
                print(f"    ... and {len(self.dependency_findings) - 5} more")
        else:
            print("    No vulnerabilities found")

        # Overall Status
        print("\n" + "-" * 60)
        if self.summary.get("passed", False):
            print("  STATUS: PASSED")
        else:
            print("  STATUS: FAILED")
        print("=" * 60)


class SecurityAuditor:
    """
    Security auditor for Zenith.

    Runs SAST and dependency vulnerability scans.
    """

    def __init__(self, project_root: Path):
        """
        Initialize auditor.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.zenith_path = project_root / "zenith"
        self.report = None

    def _get_zenith_version(self) -> str:
        """Get current Zenith version."""
        try:
            from zenith import __version__

            return __version__
        except ImportError:
            return "unknown"

    def _check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available in PATH."""
        try:
            result = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def run_bandit(self) -> list:
        """
        Run Bandit SAST scanner.

        Returns:
            List of SecurityFinding objects
        """
        findings = []

        if not self._check_tool_available("bandit"):
            print("  [WARN] Bandit not installed. Install with: pip install bandit")
            return findings

        print("  Running Bandit SAST scan...")

        try:
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    str(self.zenith_path),
                    "-f",
                    "json",
                    "-ll",  # Only medium and high severity
                    "--exclude",
                    "*/tests/*,*/__pycache__/*",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_root),
            )

            if result.stdout:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    findings.append(
                        SecurityFinding(
                            severity=issue.get("issue_severity", "UNKNOWN"),
                            category=issue.get("issue_text", "Unknown"),
                            description=issue.get("issue_text", ""),
                            file_path=issue.get("filename", ""),
                            line_number=issue.get("line_number"),
                            cwe_id=f"CWE-{issue.get('issue_cwe', {}).get('id', 'N/A')}",
                            recommendation=issue.get("more_info", ""),
                        )
                    )

        except subprocess.TimeoutExpired:
            print("  [ERROR] Bandit scan timed out")
        except json.JSONDecodeError:
            print("  [ERROR] Failed to parse Bandit output")
        except Exception as e:
            print(f"  [ERROR] Bandit scan failed: {e}")

        return findings

    def run_safety(self) -> list:
        """
        Run Safety dependency vulnerability scanner.

        Returns:
            List of SecurityFinding objects
        """
        findings = []

        if not self._check_tool_available("safety"):
            print("  [WARN] Safety not installed. Install with: pip install safety")
            return findings

        print("  Running Safety dependency scan...")

        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    vulnerabilities = data if isinstance(data, list) else []

                    for vuln in vulnerabilities:
                        if isinstance(vuln, list) and len(vuln) >= 4:
                            findings.append(
                                SecurityFinding(
                                    severity="HIGH",
                                    category="Vulnerable Dependency",
                                    description=f"{vuln[0]} {vuln[2]}: {vuln[3]}",
                                    recommendation=f"Upgrade to version > {vuln[2]}",
                                )
                            )
                except json.JSONDecodeError:
                    # Safety might output non-JSON on success
                    pass

        except subprocess.TimeoutExpired:
            print("  [ERROR] Safety scan timed out")
        except Exception as e:
            print(f"  [ERROR] Safety scan failed: {e}")

        return findings

    def run_pip_audit(self) -> list:
        """
        Run pip-audit for additional dependency checking.

        Returns:
            List of SecurityFinding objects
        """
        findings = []

        if not self._check_tool_available("pip-audit"):
            print("  [INFO] pip-audit not installed (optional)")
            return findings

        print("  Running pip-audit...")

        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for vuln in data:
                        findings.append(
                            SecurityFinding(
                                severity=vuln.get("vulns", [{}])[0].get(
                                    "fix_versions", ["HIGH"]
                                )[0]
                                if vuln.get("vulns")
                                else "HIGH",
                                category="Vulnerable Dependency",
                                description=(
                                    f"{vuln.get('name', 'Unknown')} "
                                    f"{vuln.get('version', '')}: "
                                    f"{vuln.get('vulns', [{}])[0].get('id', 'CVE-Unknown')}"
                                ),
                                cwe_id=vuln.get("vulns", [{}])[0].get("id", ""),
                            )
                        )
                except json.JSONDecodeError:
                    pass

        except subprocess.TimeoutExpired:
            print("  [ERROR] pip-audit timed out")
        except Exception as e:
            print(f"  [INFO] pip-audit: {e}")

        return findings

    def run_custom_checks(self) -> list:
        """
        Run custom security checks.

        Checks for:
        - Hardcoded secrets
        - Unsafe file operations
        - Insecure random usage

        Returns:
            List of SecurityFinding objects
        """
        findings = []
        print("  Running custom security checks...")

        # Patterns to check
        dangerous_patterns = [
            ("password\\s*=\\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            ("api_key\\s*=\\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
            ("secret\\s*=\\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            ("pickle\\.loads?\\(", "Insecure deserialization"),
            ("subprocess\\.call\\([^,]+shell\\s*=\\s*True", "Shell injection risk"),
        ]

        import re

        for py_file in self.zenith_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                for pattern, description in dangerous_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if in comment
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue

                            findings.append(
                                SecurityFinding(
                                    severity="MEDIUM",
                                    category="Custom Check",
                                    description=description,
                                    file_path=str(
                                        py_file.relative_to(self.project_root)
                                    ),
                                    line_number=i,
                                    recommendation=f"Review and fix: {description}",
                                )
                            )

            except Exception:
                continue

        return findings

    def run_audit(
        self,
        sast_only: bool = False,
        deps_only: bool = False,
    ) -> AuditReport:
        """
        Run complete security audit.

        Args:
            sast_only: Only run SAST checks
            deps_only: Only run dependency checks

        Returns:
            AuditReport with all findings
        """
        start_time = datetime.now()

        self.report = AuditReport(
            timestamp=start_time.isoformat(),
            zenith_version=self._get_zenith_version(),
            scan_duration_seconds=0.0,
        )

        print("\n" + "=" * 60)
        print("  ZENITH SECURITY AUDIT")
        print("=" * 60)

        # SAST Checks
        if not deps_only:
            print("\n[1/3] SAST SCANNING")
            print("-" * 40)

            bandit_findings = self.run_bandit()
            for f in bandit_findings:
                self.report.add_sast_finding(f)
            print(f"  Bandit: {len(bandit_findings)} findings")

            custom_findings = self.run_custom_checks()
            for f in custom_findings:
                self.report.add_sast_finding(f)
            print(f"  Custom: {len(custom_findings)} findings")

        # Dependency Checks
        if not sast_only:
            print("\n[2/3] DEPENDENCY SCANNING")
            print("-" * 40)

            safety_findings = self.run_safety()
            for f in safety_findings:
                self.report.add_dependency_finding(f)
            print(f"  Safety: {len(safety_findings)} vulnerabilities")

            pip_audit_findings = self.run_pip_audit()
            for f in pip_audit_findings:
                self.report.add_dependency_finding(f)
            print(f"  pip-audit: {len(pip_audit_findings)} vulnerabilities")

        print("\n[3/3] GENERATING REPORT")
        print("-" * 40)

        end_time = datetime.now()
        self.report.scan_duration_seconds = (end_time - start_time).total_seconds()
        self.report.compute_summary()

        return self.report


def main():
    parser = argparse.ArgumentParser(
        description="Zenith Security Audit Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sast-only",
        action="store_true",
        help="Run only SAST checks",
    )

    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Run only dependency checks",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON report to file",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory",
    )

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Default to parent of security directory
        project_root = Path(__file__).parent.parent

    if not (project_root / "zenith").exists():
        print(f"Error: zenith directory not found in {project_root}")
        return 1

    # Run audit
    auditor = SecurityAuditor(project_root)
    report = auditor.run_audit(
        sast_only=args.sast_only,
        deps_only=args.deps_only,
    )

    # Print report
    report.print_report()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report.to_json())
        print(f"\nReport saved to: {output_path}")

    # Return exit code based on findings
    if report.summary.get("passed", False):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
