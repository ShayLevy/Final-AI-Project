"""
Code-Based Graders for Insurance Claim System

Deterministic evaluation graders using exact match and regex patterns.
Based on Anthropic's "Demystifying Evals for AI Agents" recommendations.

These graders are:
- Fast, cheap, objective, reproducible, easy to debug
- Binary pass/fail scoring (0 or 1)
- No LLM calls required
"""

import re
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ground truth values extracted from insurance_claim_CLM2024001.pdf
GROUND_TRUTH = {
    # Identifiers
    "claim_id": "CLM-2024-001",
    "policy_number": "POL-2024-VEH-45782",
    "vin": "1HGCV1F39LA012345",

    # People
    "policyholder": "Sarah Mitchell",
    "at_fault_driver": "Robert Harrison",
    "claims_adjuster": "Kevin Park",

    # Dates and Times
    "incident_date": "January 12, 2024",
    "incident_time": "7:42 AM",

    # Financial
    "collision_deductible": "$750",
    "total_claim": "$23,370.80",
    "repair_cost": "$17,111.83",

    # Medical/Other
    "bac_level": "0.14%",
    "pt_sessions": "8",
}

# Alternative acceptable values (for flexibility)
GROUND_TRUTH_ALTERNATIVES = {
    "collision_deductible": ["$750", "750", "$750.00"],
    "incident_time": ["7:42 AM", "7:42:15 AM", "7:42AM"],
    "bac_level": ["0.14%", "0.14", ".14%"],
    "pt_sessions": ["8", "eight", "8 sessions"],
}

# Regex patterns for extraction and validation
REGEX_PATTERNS = {
    "claim_id": r"CLM-\d{4}-\d{3}",
    "policy_number": r"POL-\d{4}-[A-Z]{3}-\d{5}",
    "currency": r"\$[\d,]+\.\d{2}",
    "date": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    "time": r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)",
    "vin": r"[A-HJ-NPR-Z0-9]{17}",
    "phone": r"\(\d{3}\)\s*\d{3}-\d{4}",
    "percentage": r"\d+\.?\d*%",
}

# Sample texts for standalone regex validation tests
SAMPLE_TEXTS = {
    "claim_id": "The insurance claim CLM-2024-001 was filed on January 15, 2024 by Sarah Mitchell.",
    "policy_number": "Policy number POL-2024-VEH-45782 covers comprehensive and collision insurance.",
    "currency": "The total claim amount was $23,370.80 including $17,111.83 for repairs.",
    "date": "The incident occurred on January 12, 2024 at the intersection of Wilshire Blvd.",
    "time": "The collision happened at 7:42 AM when the at-fault driver ran a red light.",
    "vin": "The insured vehicle VIN is 1HGCV1F39LA012345, a 2021 Honda Accord.",
    "phone": "Contact the claims adjuster at (213) 555-0147 for more information.",
    "percentage": "Robert Harrison's BAC was measured at 0.14%, above the legal limit of 0.08%.",
}


class CodeBasedGraders:
    """
    Deterministic evaluation graders using exact match and regex patterns.

    All graders return binary pass/fail (score: 0 or 1).
    """

    @staticmethod
    def exact_match_grade(
        answer: str,
        expected: str,
        case_sensitive: bool = False,
        alternatives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check if expected value appears in answer.

        Args:
            answer: The text to search in (e.g., RAG system response)
            expected: The expected value to find
            case_sensitive: Whether to match case exactly
            alternatives: List of alternative acceptable values

        Returns:
            Dict with: passed (bool), score (0 or 1), found (str or None), details (str)
        """
        values_to_check = [expected]
        if alternatives:
            values_to_check.extend(alternatives)

        for value in values_to_check:
            if case_sensitive:
                if value in answer:
                    return {
                        "passed": True,
                        "score": 1,
                        "found": value,
                        "expected": expected,
                        "details": f"Found exact match: '{value}'"
                    }
            else:
                if value.lower() in answer.lower():
                    return {
                        "passed": True,
                        "score": 1,
                        "found": value,
                        "expected": expected,
                        "details": f"Found match (case-insensitive): '{value}'"
                    }

        return {
            "passed": False,
            "score": 0,
            "found": None,
            "expected": expected,
            "details": f"Expected '{expected}' not found in answer"
        }

    @staticmethod
    def regex_grade(
        answer: str,
        pattern: str,
        expected_value: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract pattern from answer, optionally validate against expected value.

        Args:
            answer: The text to search in
            pattern: Regex pattern to match
            expected_value: If provided, check if this specific value is among matches

        Returns:
            Dict with: passed (bool), score (0 or 1), matches (list), details (str)
        """
        try:
            matches = re.findall(pattern, answer, re.IGNORECASE)

            if expected_value:
                # Check if expected value is among matches
                passed = any(
                    expected_value.lower() == m.lower()
                    for m in matches
                )
                details = f"Looking for '{expected_value}' in matches: {matches}"
            else:
                # Just check if pattern matched anything
                passed = len(matches) > 0
                details = f"Pattern matched {len(matches)} time(s): {matches[:5]}"  # Limit to first 5

            return {
                "passed": passed,
                "score": 1 if passed else 0,
                "matches": matches,
                "pattern": pattern,
                "expected_value": expected_value,
                "details": details
            }

        except re.error as e:
            return {
                "passed": False,
                "score": 0,
                "matches": [],
                "pattern": pattern,
                "expected_value": expected_value,
                "details": f"Regex error: {str(e)}"
            }

    @staticmethod
    def run_rag_test(
        answer: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single test case against a RAG system response.

        Args:
            answer: RAG system response text
            test_case: Test case dictionary with grader_type, expected_value, etc.

        Returns:
            Grading result dictionary
        """
        grader_type = test_case.get("grader_type", "exact_match")
        expected_value = test_case.get("expected_value")

        if grader_type == "exact_match":
            alternatives = GROUND_TRUTH_ALTERNATIVES.get(
                test_case.get("ground_truth_key"), []
            )
            result = CodeBasedGraders.exact_match_grade(
                answer=answer,
                expected=expected_value,
                case_sensitive=test_case.get("case_sensitive", False),
                alternatives=alternatives
            )
        elif grader_type == "regex":
            pattern = test_case.get("regex_pattern") or REGEX_PATTERNS.get(
                test_case.get("pattern_name", "")
            )
            result = CodeBasedGraders.regex_grade(
                answer=answer,
                pattern=pattern,
                expected_value=expected_value
            )
        else:
            result = {
                "passed": False,
                "score": 0,
                "details": f"Unknown grader type: {grader_type}"
            }

        # Add test case metadata to result
        result["test_id"] = test_case.get("id", "unknown")
        result["query"] = test_case.get("query", "")
        result["grader_type"] = grader_type

        return result

    @staticmethod
    def run_standalone_regex_test(pattern_name: str) -> Dict[str, Any]:
        """
        Test a regex pattern against known sample text.

        Args:
            pattern_name: Key in REGEX_PATTERNS dict

        Returns:
            Test result dictionary
        """
        if pattern_name not in REGEX_PATTERNS:
            return {
                "passed": False,
                "score": 0,
                "pattern_name": pattern_name,
                "details": f"Unknown pattern: {pattern_name}"
            }

        pattern = REGEX_PATTERNS[pattern_name]
        sample_text = SAMPLE_TEXTS.get(pattern_name, "")

        if not sample_text:
            return {
                "passed": False,
                "score": 0,
                "pattern_name": pattern_name,
                "details": f"No sample text for pattern: {pattern_name}"
            }

        result = CodeBasedGraders.regex_grade(
            answer=sample_text,
            pattern=pattern
        )

        result["pattern_name"] = pattern_name
        result["sample_text"] = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text

        return result

    @staticmethod
    def run_all_standalone_tests() -> List[Dict[str, Any]]:
        """
        Run all standalone regex validation tests.

        Returns:
            List of test results
        """
        results = []
        for pattern_name in REGEX_PATTERNS.keys():
            result = CodeBasedGraders.run_standalone_regex_test(pattern_name)
            results.append(result)
        return results

    @staticmethod
    def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics from a list of test results.

        Args:
            results: List of individual test results

        Returns:
            Summary dictionary with pass rate, totals, etc.
        """
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        failed = total - passed

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "score": passed,  # Binary scoring, just count passes
            "max_score": total
        }

    @staticmethod
    def run_self_tests() -> bool:
        """
        Run self-tests to verify graders work correctly.

        Returns:
            True if all tests pass, False otherwise
        """
        print("Running Code-Based Graders self-tests...")

        all_passed = True

        # Test 1: Exact match - should pass
        result = CodeBasedGraders.exact_match_grade(
            answer="The claim ID is CLM-2024-001 for Sarah Mitchell.",
            expected="CLM-2024-001"
        )
        if not result["passed"]:
            print(f"FAIL: Test 1 - Exact match should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 1 - Exact match")

        # Test 2: Exact match - should fail
        result = CodeBasedGraders.exact_match_grade(
            answer="The claim ID is CLM-2024-002.",
            expected="CLM-2024-001"
        )
        if result["passed"]:
            print(f"FAIL: Test 2 - Exact match should fail: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 2 - Exact match (correctly failed)")

        # Test 3: Regex extraction - should find matches
        result = CodeBasedGraders.regex_grade(
            answer="The total was $23,370.80 and repairs cost $17,111.83.",
            pattern=REGEX_PATTERNS["currency"]
        )
        if not result["passed"] or len(result["matches"]) != 2:
            print(f"FAIL: Test 3 - Should find 2 currency matches: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 3 - Regex currency extraction")

        # Test 4: Regex with expected value - should validate
        result = CodeBasedGraders.regex_grade(
            answer="The claim CLM-2024-001 was filed.",
            pattern=REGEX_PATTERNS["claim_id"],
            expected_value="CLM-2024-001"
        )
        if not result["passed"]:
            print(f"FAIL: Test 4 - Should validate expected claim ID: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 4 - Regex with validation")

        # Test 5: Standalone regex tests
        standalone_results = CodeBasedGraders.run_all_standalone_tests()
        summary = CodeBasedGraders.calculate_summary(standalone_results)
        if summary["pass_rate"] < 100:
            print(f"FAIL: Test 5 - Standalone tests: {summary}")
            all_passed = False
        else:
            print(f"PASS: Test 5 - All standalone regex tests ({summary['passed']}/{summary['total_tests']})")

        print(f"\nSelf-test result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        return all_passed


if __name__ == "__main__":
    # Run self-tests when module is executed directly
    CodeBasedGraders.run_self_tests()
