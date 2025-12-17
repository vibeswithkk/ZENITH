#!/usr/bin/env python3
"""
Mutation Testing Script for Zenith

Runs mutation testing on critical modules with target 82% mutation score.
Based on CetakBiru Section 5.3 requirements.
"""

import subprocess
import tempfile
import shutil
import ast
import os
import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MutationResult:
    """Result of a mutation test."""

    file: str
    line: int
    original: str
    mutated: str
    killed: bool
    error: str = ""


class SimpleMutator:
    """Simple Python mutation testing engine."""

    MUTATIONS = [
        # Comparison operator swaps
        ("==", "!="),
        ("!=", "=="),
        (">", "<"),
        ("<", ">"),
        (">=", "<="),
        ("<=", ">="),
        # Arithmetic operator swaps
        ("+", "-"),
        ("-", "+"),
        ("*", "/"),
        ("/", "*"),
        # Boolean swaps
        ("True", "False"),
        ("False", "True"),
        ("and", "or"),
        ("or", "and"),
        ("not ", ""),
        # Return value mutations
        ("return 0", "return 1"),
        ("return 1", "return 0"),
        ("return True", "return False"),
        ("return False", "return True"),
        ("return None", "return 0"),
    ]

    def __init__(self, test_command: List[str]):
        self.test_command = test_command
        self.results: List[MutationResult] = []

    def generate_mutants(self, source_file: str) -> List[Tuple[int, str, str]]:
        """Generate mutants for a source file."""
        with open(source_file, "r") as f:
            lines = f.readlines()

        mutants = []
        for i, line in enumerate(lines, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue

            for original, replacement in self.MUTATIONS:
                if original in line and not line.strip().startswith("#"):
                    mutated = line.replace(original, replacement, 1)
                    if mutated != line:
                        mutants.append((i, line.rstrip(), mutated.rstrip()))

        return mutants

    def run_tests(self) -> bool:
        """Run test suite and return True if tests pass."""
        result = subprocess.run(
            self.test_command,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0

    def test_mutant(
        self, source_file: str, line_num: int, original: str, mutated: str
    ) -> MutationResult:
        """Test a single mutant."""
        # Backup original file
        with open(source_file, "r") as f:
            original_content = f.read()

        # Apply mutation
        lines = original_content.split("\n")
        lines[line_num - 1] = mutated

        try:
            with open(source_file, "w") as f:
                f.write("\n".join(lines))

            # Run tests
            killed = not self.run_tests()

            return MutationResult(
                file=source_file,
                line=line_num,
                original=original,
                mutated=mutated,
                killed=killed,
            )
        except subprocess.TimeoutExpired:
            return MutationResult(
                file=source_file,
                line=line_num,
                original=original,
                mutated=mutated,
                killed=True,  # Timeout = killed
                error="timeout",
            )
        except Exception as e:
            return MutationResult(
                file=source_file,
                line=line_num,
                original=original,
                mutated=mutated,
                killed=True,
                error=str(e),
            )
        finally:
            # Restore original file
            with open(source_file, "w") as f:
                f.write(original_content)

    def run_mutation_testing(self, source_file: str, max_mutants: int = 50) -> float:
        """Run mutation testing on a source file."""
        print(f"\n{'=' * 60}")
        print(f"Mutation Testing: {source_file}")
        print(f"{'=' * 60}")

        # Generate mutants
        mutants = self.generate_mutants(source_file)
        print(f"Generated {len(mutants)} potential mutants")

        # Sample if too many
        if len(mutants) > max_mutants:
            mutants = random.sample(mutants, max_mutants)
            print(f"Sampling {max_mutants} mutants for testing")

        # Test each mutant
        killed = 0
        survived = 0
        errors = 0

        for i, (line_num, original, mutated) in enumerate(mutants, 1):
            result = self.test_mutant(source_file, line_num, original, mutated)
            self.results.append(result)

            if result.error:
                status = "ERROR"
                errors += 1
            elif result.killed:
                status = "KILLED"
                killed += 1
            else:
                status = "SURVIVED"
                survived += 1

            print(f"  [{i}/{len(mutants)}] Line {line_num}: {status}")
            if not result.killed and not result.error:
                print(f"       Original: {original[:60]}")
                print(f"       Mutated:  {mutated[:60]}")

        # Calculate score
        total = killed + survived
        if total == 0:
            return 100.0

        score = (killed / total) * 100

        print(f"\n{'=' * 60}")
        print(f"Results for {source_file}:")
        print(f"  Killed:   {killed}")
        print(f"  Survived: {survived}")
        print(f"  Errors:   {errors}")
        print(f"  Score:    {score:.1f}%")
        print(f"{'=' * 60}")

        return score


def main():
    """Run mutation testing on critical modules."""
    # Critical modules to test
    critical_modules = [
        "zenith/core/types.py",
        "zenith/core/graph_ir.py",
        "zenith/core/tensor.py",
        "zenith/core/node.py",
    ]

    # Test command (without coverage)
    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_core_unit.py",
        "-x",
        "-q",
        "--no-cov",
    ]

    mutator = SimpleMutator(test_command)

    total_killed = 0
    total_survived = 0
    module_scores = {}

    for module in critical_modules:
        if os.path.exists(module):
            score = mutator.run_mutation_testing(module, max_mutants=30)
            module_scores[module] = score

            # Count results for this module
            for result in mutator.results:
                if result.file == module:
                    if result.killed:
                        total_killed += 1
                    else:
                        total_survived += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print("MUTATION TESTING SUMMARY")
    print(f"{'=' * 60}")

    for module, score in module_scores.items():
        status = "PASS" if score >= 82 else "FAIL"
        print(f"  {module}: {score:.1f}% [{status}]")

    total = total_killed + total_survived
    if total > 0:
        overall_score = (total_killed / total) * 100
    else:
        overall_score = 100.0

    print(f"\nOverall Score: {overall_score:.1f}%")
    print(f"Target: 82%")

    if overall_score >= 82:
        print("STATUS: PASSED")
        return 0
    else:
        print("STATUS: NEEDS IMPROVEMENT")
        return 1


if __name__ == "__main__":
    sys.exit(main())
