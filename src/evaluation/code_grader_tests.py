"""
Test Cases for Code-Based Graders

Two test modes:
1. RAG Response Grading - Query the RAG system, then grade the response
2. Standalone Regex Validation - Test regex patterns against sample text (no RAG)

Ground truth derived from: data/insurance_claim_CLM2024001.pdf
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeGraderTestSuite:
    """
    Test suite for code-based graders.
    Contains both RAG response tests and standalone regex validation tests.
    """

    @staticmethod
    def get_rag_test_cases() -> List[Dict[str, Any]]:
        """
        Get test cases for RAG response grading.
        These tests query the RAG system and grade the response.

        Returns:
            List of test case dictionaries
        """
        return [
            # Test 1: Claim ID
            {
                "id": "CBG_RAG_01",
                "query": "What is the claim ID?",
                "grader_type": "exact_match",
                "expected_value": "CLM-2024-001",
                "ground_truth_key": "claim_id",
                "case_sensitive": False,
                "description": "Verify claim ID extraction",
                "category": "identifiers"
            },

            # Test 2: Policyholder
            {
                "id": "CBG_RAG_02",
                "query": "Who is the policyholder?",
                "grader_type": "exact_match",
                "expected_value": "Sarah Mitchell",
                "ground_truth_key": "policyholder",
                "case_sensitive": False,
                "description": "Verify policyholder name extraction",
                "category": "people"
            },

            # Test 3: Collision Deductible
            {
                "id": "CBG_RAG_03",
                "query": "What was the collision deductible amount?",
                "grader_type": "exact_match",
                "expected_value": "$750",
                "ground_truth_key": "collision_deductible",
                "case_sensitive": False,
                "description": "Verify collision deductible extraction",
                "category": "financial"
            },

            # Test 4: Incident Date
            {
                "id": "CBG_RAG_04",
                "query": "When did the accident occur? What was the date?",
                "grader_type": "exact_match",
                "expected_value": "January 12, 2024",
                "ground_truth_key": "incident_date",
                "case_sensitive": False,
                "description": "Verify incident date extraction",
                "category": "dates"
            },

            # Test 5: Total Claim Amount
            {
                "id": "CBG_RAG_05",
                "query": "What was the total claim amount?",
                "grader_type": "exact_match",
                "expected_value": "$23,370.80",
                "ground_truth_key": "total_claim",
                "case_sensitive": False,
                "description": "Verify total claim amount extraction",
                "category": "financial"
            },

            # Test 6: At-fault Driver
            {
                "id": "CBG_RAG_06",
                "query": "Who was the at-fault driver?",
                "grader_type": "exact_match",
                "expected_value": "Robert Harrison",
                "ground_truth_key": "at_fault_driver",
                "case_sensitive": False,
                "description": "Verify at-fault driver name extraction",
                "category": "people"
            },

            # Test 7: BAC Level
            {
                "id": "CBG_RAG_07",
                "query": "What was the at-fault driver's blood alcohol concentration (BAC)?",
                "grader_type": "exact_match",
                "expected_value": "0.14%",
                "ground_truth_key": "bac_level",
                "case_sensitive": False,
                "description": "Verify BAC level extraction",
                "category": "medical"
            },

            # Test 8: Claims Adjuster
            {
                "id": "CBG_RAG_08",
                "query": "Who is the claims adjuster assigned to this case?",
                "grader_type": "exact_match",
                "expected_value": "Kevin Park",
                "ground_truth_key": "claims_adjuster",
                "case_sensitive": False,
                "description": "Verify claims adjuster name extraction",
                "category": "people"
            },

            # Test 9: Physical Therapy Sessions
            {
                "id": "CBG_RAG_09",
                "query": "How many physical therapy sessions did the policyholder complete?",
                "grader_type": "exact_match",
                "expected_value": "8",
                "ground_truth_key": "pt_sessions",
                "case_sensitive": False,
                "description": "Verify PT session count extraction",
                "category": "medical"
            },

            # Test 10: Repair Cost
            {
                "id": "CBG_RAG_10",
                "query": "What was the vehicle repair cost?",
                "grader_type": "exact_match",
                "expected_value": "$17,111.83",
                "ground_truth_key": "repair_cost",
                "case_sensitive": False,
                "description": "Verify repair cost extraction",
                "category": "financial"
            },
        ]

    @staticmethod
    def get_regex_test_cases() -> List[Dict[str, Any]]:
        """
        Get test cases for standalone regex validation.
        These tests validate that regex patterns work correctly.

        Returns:
            List of test case dictionaries
        """
        return [
            # Test 1: Claim ID Pattern
            {
                "id": "CBG_REGEX_01",
                "pattern_name": "claim_id",
                "regex_pattern": r"CLM-\d{4}-\d{3}",
                "expected_matches": ["CLM-2024-001"],
                "description": "Validate claim ID regex pattern (CLM-YYYY-NNN)",
                "category": "identifiers"
            },

            # Test 2: Currency Pattern
            {
                "id": "CBG_REGEX_02",
                "pattern_name": "currency",
                "regex_pattern": r"\$[\d,]+\.\d{2}",
                "expected_matches": ["$23,370.80", "$17,111.83"],
                "description": "Validate currency regex pattern ($X,XXX.XX)",
                "category": "financial"
            },

            # Test 3: Date Pattern
            {
                "id": "CBG_REGEX_03",
                "pattern_name": "date",
                "regex_pattern": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
                "expected_matches": ["January 12, 2024"],
                "description": "Validate date regex pattern (Month DD, YYYY)",
                "category": "dates"
            },

            # Test 4: Time Pattern
            {
                "id": "CBG_REGEX_04",
                "pattern_name": "time",
                "regex_pattern": r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)",
                "expected_matches": ["7:42 AM"],
                "description": "Validate time regex pattern (HH:MM AM/PM)",
                "category": "dates"
            },

            # Test 5: VIN Pattern
            {
                "id": "CBG_REGEX_05",
                "pattern_name": "vin",
                "regex_pattern": r"[A-HJ-NPR-Z0-9]{17}",
                "expected_matches": ["1HGCV1F39LA012345"],
                "description": "Validate VIN regex pattern (17 alphanumeric)",
                "category": "identifiers"
            },

            # Test 6: Phone Pattern
            {
                "id": "CBG_REGEX_06",
                "pattern_name": "phone",
                "regex_pattern": r"\(\d{3}\)\s*\d{3}-\d{4}",
                "expected_matches": ["(213) 555-0147"],
                "description": "Validate phone regex pattern ((XXX) XXX-XXXX)",
                "category": "contact"
            },

            # Test 7: Percentage Pattern
            {
                "id": "CBG_REGEX_07",
                "pattern_name": "percentage",
                "regex_pattern": r"\d+\.?\d*%",
                "expected_matches": ["0.14%", "0.08%"],
                "description": "Validate percentage regex pattern (X.XX%)",
                "category": "medical"
            },

            # Test 8: Policy Number Pattern
            {
                "id": "CBG_REGEX_08",
                "pattern_name": "policy_number",
                "regex_pattern": r"POL-\d{4}-[A-Z]{3}-\d{5}",
                "expected_matches": ["POL-2024-VEH-45782"],
                "description": "Validate policy number regex pattern (POL-YYYY-XXX-NNNNN)",
                "category": "identifiers"
            },
        ]

    @staticmethod
    def get_all_test_cases() -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all test cases organized by type.

        Returns:
            Dictionary with 'rag' and 'regex' test case lists
        """
        return {
            "rag": CodeGraderTestSuite.get_rag_test_cases(),
            "regex": CodeGraderTestSuite.get_regex_test_cases()
        }

    @staticmethod
    def get_test_by_id(test_id: str) -> Dict[str, Any]:
        """
        Get a specific test case by ID.

        Args:
            test_id: Test case ID (e.g., "CBG_RAG_01")

        Returns:
            Test case dictionary or empty dict if not found
        """
        all_cases = CodeGraderTestSuite.get_all_test_cases()

        for test_type, cases in all_cases.items():
            for case in cases:
                if case["id"] == test_id:
                    case["test_type"] = test_type
                    return case

        return {}

    @staticmethod
    def get_tests_by_category(category: str) -> List[Dict[str, Any]]:
        """
        Get all test cases in a specific category.

        Args:
            category: Category name (identifiers, people, financial, dates, medical, contact)

        Returns:
            List of matching test cases
        """
        all_cases = CodeGraderTestSuite.get_all_test_cases()
        results = []

        for test_type, cases in all_cases.items():
            for case in cases:
                if case.get("category") == category:
                    case_copy = case.copy()
                    case_copy["test_type"] = test_type
                    results.append(case_copy)

        return results

    @staticmethod
    def print_summary():
        """Print summary of available test cases."""
        all_cases = CodeGraderTestSuite.get_all_test_cases()

        print("\n" + "=" * 70)
        print("CODE-BASED GRADER TEST SUITE")
        print("=" * 70)

        for test_type, cases in all_cases.items():
            print(f"\n{test_type.upper()} Tests ({len(cases)} tests):")
            print("-" * 50)

            for case in cases:
                print(f"  [{case['id']}] {case['description']}")
                if test_type == "rag":
                    print(f"      Query: {case['query'][:50]}...")
                    print(f"      Expected: {case['expected_value']}")
                else:
                    print(f"      Pattern: {case['regex_pattern'][:40]}...")

        # Category summary
        print("\n" + "-" * 50)
        print("Categories:")
        categories = {}
        for test_type, cases in all_cases.items():
            for case in cases:
                cat = case.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} tests")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    CodeGraderTestSuite.print_summary()
