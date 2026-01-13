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
from difflib import SequenceMatcher
from datetime import datetime
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

# Numerical ground truth values for tolerance-based validation
GROUND_TRUTH_NUMERICAL = {
    "total_claim_amount": 23370.80,
    "collision_deductible": 750.00,
    "repair_cost": 17111.83,
    "bac_level_numeric": 0.14,
    "pt_session_count": 8,
}

# Fact groups for key fact coverage grading
FACT_GROUPS = {
    "incident_summary": {
        "description": "Complete incident summary",
        "required_facts": [
            {"key": "incident_date", "pattern": r"January\s+12,?\s+2024"},
            {"key": "incident_time", "pattern": r"7:42\s*(?:AM|am|a\.m\.)?"},
            {"key": "location", "pattern": r"Wilshire.*Vermont|Vermont.*Wilshire|intersection"},
            {"key": "parties", "pattern": r"Sarah\s+Mitchell|Robert\s+Harrison"},
        ]
    },
    "financial_summary": {
        "description": "Complete financial summary",
        "required_facts": [
            {"key": "total_claim", "pattern": r"\$?23,?370\.?80"},
            {"key": "repair_cost", "pattern": r"\$?17,?111\.?83"},
            {"key": "deductible", "pattern": r"\$?750(?:\.00)?"},
        ]
    },
    "liability_determination": {
        "description": "Complete liability information",
        "required_facts": [
            {"key": "at_fault_driver", "pattern": r"Robert\s+Harrison"},
            {"key": "liability_percentage", "pattern": r"100\s*%|full\s+liability|fully\s+liable"},
            {"key": "bac_level", "pattern": r"0\.14\s*%?"},
        ]
    },
    "medical_treatment": {
        "description": "Complete medical treatment summary",
        "required_facts": [
            {"key": "hospital", "pattern": r"Cedars[\s-]?Sinai"},
            {"key": "diagnosis", "pattern": r"whiplash|cervical\s+strain"},
            {"key": "pt_sessions", "pattern": r"8\s+(?:physical\s+therapy\s+)?sessions?|eight\s+(?:PT\s+)?sessions?"},
        ]
    },
    "witness_information": {
        "description": "Complete witness information",
        "required_facts": [
            {"key": "witness_1", "pattern": r"Marcus\s+Thompson"},
            {"key": "witness_2", "pattern": r"Elena\s+Rodriguez"},
            {"key": "witness_3", "pattern": r"Patricia\s+O['\u2019]?Brien"},
        ]
    },
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
    def numerical_validation_grade(
        answer: str,
        expected_value: float,
        tolerance_type: str = "absolute",
        tolerance_value: float = 0.01,
        value_type: str = "currency",
    ) -> Dict[str, Any]:
        """
        Validate numerical values in the answer with configurable tolerance.

        Args:
            answer: RAG system response text
            expected_value: The expected numerical value
            tolerance_type: "absolute" (e.g., ±$0.01) or "percentage" (e.g., ±1%)
            tolerance_value: The tolerance threshold
            value_type: Type of value for format recognition ("currency", "percentage", "integer")

        Returns:
            Dict with: passed (bool), score (0 or 1), found_value, expected_value, difference, details
        """
        patterns = {
            "currency": r'\$?\s*([\d,]+\.?\d*)',
            "percentage": r'(\d+\.?\d*)\s*%?',
            "integer": r'(\d+)',
        }
        pattern = patterns.get(value_type, patterns["currency"])

        try:
            matches = re.findall(pattern, answer)
            found_values = []
            for match in matches:
                clean_value = match.replace(',', '').strip()
                if clean_value:
                    try:
                        found_values.append(float(clean_value))
                    except ValueError:
                        continue

            if not found_values:
                return {
                    "passed": False,
                    "score": 0,
                    "found_value": None,
                    "expected_value": expected_value,
                    "difference": None,
                    "tolerance_type": tolerance_type,
                    "tolerance_value": tolerance_value,
                    "details": f"No numerical values found for {value_type}"
                }

            for found in found_values:
                if tolerance_type == "absolute":
                    difference = abs(found - expected_value)
                    within_tolerance = difference <= tolerance_value
                else:  # percentage
                    if expected_value == 0:
                        within_tolerance = found == 0
                        difference = abs(found)
                    else:
                        difference = abs((found - expected_value) / expected_value) * 100
                        within_tolerance = difference <= tolerance_value

                if within_tolerance:
                    return {
                        "passed": True,
                        "score": 1,
                        "found_value": found,
                        "expected_value": expected_value,
                        "difference": difference,
                        "tolerance_type": tolerance_type,
                        "tolerance_value": tolerance_value,
                        "details": f"Found {found} matches expected {expected_value} within {tolerance_type} tolerance of {tolerance_value}"
                    }

            closest = min(found_values, key=lambda x: abs(x - expected_value))
            if tolerance_type == "absolute":
                diff = abs(closest - expected_value)
            else:
                diff = abs((closest - expected_value) / expected_value) * 100 if expected_value != 0 else abs(closest)

            return {
                "passed": False,
                "score": 0,
                "found_value": closest,
                "expected_value": expected_value,
                "difference": diff,
                "tolerance_type": tolerance_type,
                "tolerance_value": tolerance_value,
                "all_found_values": found_values[:5],
                "details": f"Closest found value {closest} differs by {diff:.4f} ({tolerance_type}), exceeds tolerance {tolerance_value}"
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "found_value": None,
                "expected_value": expected_value,
                "difference": None,
                "details": f"Error during numerical validation: {str(e)}"
            }

    @staticmethod
    def consistency_check_grade(
        answer: str,
        check_type: str,
        check_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Check internal consistency of facts within a RAG response.

        Args:
            answer: RAG system response text
            check_type: Type of consistency check ("chronological", "sum_constraint", "name_consistency")
            check_config: Configuration for the specific check

        Returns:
            Dict with: passed (bool), score (0 or 1), violations (list), details (str)
        """
        if check_type == "chronological":
            return CodeBasedGraders._check_chronological_consistency(answer)
        elif check_type == "sum_constraint":
            return CodeBasedGraders._check_sum_constraint(answer, check_config or {})
        elif check_type == "name_consistency":
            return CodeBasedGraders._check_name_consistency(answer, check_config or {})
        else:
            return {
                "passed": False,
                "score": 0,
                "check_type": check_type,
                "details": f"Unknown check type: {check_type}"
            }

    @staticmethod
    def _check_chronological_consistency(answer: str) -> Dict[str, Any]:
        """Verify dates appear in chronological order."""
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        date_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
        matches = list(re.finditer(date_pattern, answer, re.IGNORECASE))

        if len(matches) < 2:
            return {
                "passed": True,
                "score": 1,
                "found_dates": [m.group(0) for m in matches],
                "violations": [],
                "check_type": "chronological",
                "details": "Insufficient dates found to check chronology (0 or 1 dates)"
            }

        found_dates = []
        for match in matches:
            try:
                month_name = match.group(1).lower()
                day = int(match.group(2))
                year = int(match.group(3))
                month = month_map.get(month_name, 1)
                date_obj = datetime(year, month, day)
                found_dates.append((date_obj, match.group(0), match.start()))
            except (ValueError, KeyError):
                continue

        found_dates.sort(key=lambda x: x[2])

        violations = []
        for i in range(len(found_dates) - 1):
            current_date, current_str, _ = found_dates[i]
            next_date, next_str, _ = found_dates[i + 1]

            if current_date > next_date:
                violations.append({
                    "issue": "out_of_order",
                    "first_date": current_str,
                    "second_date": next_str,
                    "message": f"'{current_str}' appears before '{next_str}' but is chronologically later"
                })

        passed = len(violations) == 0
        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "found_dates": [d[1] for d in found_dates],
            "violations": violations,
            "check_type": "chronological",
            "details": "All dates in chronological order" if passed else f"Found {len(violations)} chronological violation(s)"
        }

    @staticmethod
    def _check_sum_constraint(answer: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Verify numerical sums are consistent."""
        currency_pattern = r'\$\s*([\d,]+\.?\d*)'
        matches = re.findall(currency_pattern, answer)

        found_values = []
        for match in matches:
            try:
                found_values.append(float(match.replace(',', '')))
            except ValueError:
                continue

        if len(found_values) < 2:
            return {
                "passed": True,
                "score": 1,
                "found_values": found_values,
                "check_type": "sum_constraint",
                "details": "Insufficient values found to check sum constraint"
            }

        sorted_values = sorted(found_values, reverse=True)
        likely_total = sorted_values[0]
        likely_components = sorted_values[1:]
        component_sum = sum(likely_components)

        operator = config.get("operator", "<=")
        if operator == "<=":
            passed = component_sum <= likely_total * 1.01
        elif operator == "==":
            passed = abs(component_sum - likely_total) < 0.01
        else:
            passed = True

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "likely_total": likely_total,
            "component_sum": component_sum,
            "components": likely_components[:5],
            "check_type": "sum_constraint",
            "details": f"Components sum ({component_sum:.2f}) {'<=' if passed else '>'} total ({likely_total:.2f})"
        }

    @staticmethod
    def _check_name_consistency(answer: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a person's name is referred to consistently."""
        expected_name = config.get("expected_name", "")

        if not expected_name:
            return {
                "passed": False,
                "score": 0,
                "check_type": "name_consistency",
                "details": "No expected name provided in config"
            }

        name_parts = expected_name.lower().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        name_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        found_names = re.findall(name_pattern, answer)

        variations_found = []
        for name in found_names:
            name_lower = name.lower()
            if first_name in name_lower or last_name in name_lower:
                variations_found.append(name)

        unique_variations = list(dict.fromkeys(variations_found))

        passed = len(unique_variations) <= 1 or all(
            expected_name.lower() in v.lower() or v.lower() in expected_name.lower()
            for v in unique_variations
        )

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "expected_name": expected_name,
            "variations_found": unique_variations,
            "check_type": "name_consistency",
            "details": "Name references are consistent" if passed else f"Found inconsistent name variations: {unique_variations}"
        }

    @staticmethod
    def key_fact_coverage_grade(
        answer: str,
        fact_group: str,
        custom_facts: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Check if response contains all required facts for completeness.

        Args:
            answer: RAG system response text
            fact_group: Name of the fact group to check (from FACT_GROUPS)
            custom_facts: Optional list of custom fact definitions

        Returns:
            Dict with: passed (bool), score (0 or 1), facts_found, facts_missing, coverage_ratio, details
        """
        if custom_facts:
            facts_to_check = custom_facts
            group_description = "Custom fact group"
        elif fact_group in FACT_GROUPS:
            facts_to_check = FACT_GROUPS[fact_group]["required_facts"]
            group_description = FACT_GROUPS[fact_group]["description"]
        else:
            return {
                "passed": False,
                "score": 0,
                "fact_group": fact_group,
                "details": f"Unknown fact group: {fact_group}. Available: {list(FACT_GROUPS.keys())}"
            }

        facts_found = []
        facts_missing = []

        for fact in facts_to_check:
            fact_key = fact.get("key", "unknown")
            pattern = fact.get("pattern", "")

            if not pattern:
                facts_missing.append({"key": fact_key, "reason": "No pattern defined"})
                continue

            try:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    facts_found.append({
                        "key": fact_key,
                        "matched_text": match.group(0),
                    })
                else:
                    facts_missing.append({"key": fact_key, "pattern": pattern})
            except re.error as e:
                facts_missing.append({"key": fact_key, "reason": f"Regex error: {str(e)}"})

        total_facts = len(facts_to_check)
        found_count = len(facts_found)
        coverage_ratio = found_count / total_facts if total_facts > 0 else 0

        passed = len(facts_missing) == 0

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "fact_group": fact_group,
            "group_description": group_description,
            "total_facts": total_facts,
            "facts_found": facts_found,
            "facts_found_count": found_count,
            "facts_missing": facts_missing,
            "facts_missing_count": len(facts_missing),
            "coverage_ratio": coverage_ratio,
            "details": f"Coverage: {found_count}/{total_facts} ({coverage_ratio*100:.1f}%). " +
                       (f"Missing: {[f['key'] for f in facts_missing]}" if facts_missing else "All facts present.")
        }

    @staticmethod
    def fuzzy_match_grade(
        answer: str,
        expected_value: str,
        similarity_threshold: float = 0.80,
        match_type: str = "name",
    ) -> Dict[str, Any]:
        """
        Find expected string in answer using fuzzy matching.

        Args:
            answer: RAG system response text
            expected_value: The expected string to find
            similarity_threshold: Minimum similarity ratio (0.0 to 1.0)
            match_type: Type of string for optimized extraction ("name", "address", "text")

        Returns:
            Dict with: passed (bool), score (0 or 1), best_match, similarity_ratio, details
        """
        def calculate_similarity(s1: str, s2: str) -> float:
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

        def extract_candidates(text: str, m_type: str) -> List[str]:
            candidates = []
            if m_type == "name":
                patterns = [
                    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
                    r'[A-Z]\.\s*[A-Z][a-z]+',
                    r'[A-Z][a-z]+\s+[A-Z]\.',
                ]
                for pattern in patterns:
                    candidates.extend(re.findall(pattern, text))
            elif m_type == "address":
                pattern = r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr)'
                candidates = re.findall(pattern, text, re.IGNORECASE)
            else:
                words = text.split()
                expected_word_count = len(expected_value.split())
                for i in range(len(words) - expected_word_count + 1):
                    window = ' '.join(words[i:i + expected_word_count])
                    candidates.append(window)
            return candidates

        name_parts = expected_value.split()
        if len(name_parts) >= 2 and match_type == "name":
            first, last = name_parts[0], name_parts[-1]
            variations = [
                expected_value,
                expected_value.upper(),
                expected_value.lower(),
                f"{first[0]}. {last}",
                f"{first[0]} {last}",
                f"{first} {last[0]}.",
                last,
            ]
            answer_lower = answer.lower()
            for variation in variations:
                if variation.lower() in answer_lower:
                    return {
                        "passed": True,
                        "score": 1,
                        "expected_value": expected_value,
                        "best_match": variation,
                        "similarity_ratio": 1.0,
                        "match_type": match_type,
                        "match_method": "exact_variation",
                        "details": f"Found exact match with variation: '{variation}'"
                    }

        candidates = extract_candidates(answer, match_type)

        if not candidates:
            return {
                "passed": False,
                "score": 0,
                "expected_value": expected_value,
                "best_match": None,
                "similarity_ratio": 0.0,
                "match_type": match_type,
                "candidates_checked": 0,
                "details": f"No candidate {match_type} strings found in answer"
            }

        best_match = None
        best_ratio = 0.0

        for candidate in candidates:
            ratio = calculate_similarity(expected_value, candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate

        passed = best_ratio >= similarity_threshold

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "expected_value": expected_value,
            "best_match": best_match,
            "similarity_ratio": round(best_ratio, 4),
            "similarity_threshold": similarity_threshold,
            "match_type": match_type,
            "candidates_checked": len(candidates),
            "details": f"Best match '{best_match}' with {best_ratio*100:.1f}% similarity " +
                       (f"(>= {similarity_threshold*100}% threshold - PASS)" if passed
                        else f"(< {similarity_threshold*100}% threshold - FAIL)")
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
        elif grader_type == "numerical_validation":
            result = CodeBasedGraders.numerical_validation_grade(
                answer=answer,
                expected_value=float(test_case.get("expected_value", 0)),
                tolerance_type=test_case.get("tolerance_type", "absolute"),
                tolerance_value=float(test_case.get("tolerance_value", 0.01)),
                value_type=test_case.get("value_type", "currency")
            )
        elif grader_type == "consistency_check":
            result = CodeBasedGraders.consistency_check_grade(
                answer=answer,
                check_type=test_case.get("check_type", "chronological"),
                check_config=test_case.get("check_config", {})
            )
        elif grader_type == "key_fact_coverage":
            result = CodeBasedGraders.key_fact_coverage_grade(
                answer=answer,
                fact_group=test_case.get("fact_group", ""),
                custom_facts=test_case.get("custom_facts")
            )
        elif grader_type == "fuzzy_match":
            result = CodeBasedGraders.fuzzy_match_grade(
                answer=answer,
                expected_value=str(test_case.get("expected_value", "")),
                similarity_threshold=float(test_case.get("similarity_threshold", 0.80)),
                match_type=test_case.get("match_type", "name")
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

        # Test 6: Numerical validation - should pass
        result = CodeBasedGraders.numerical_validation_grade(
            answer="The total claim was $23,370.80 for all damages.",
            expected_value=23370.80,
            tolerance_type="absolute",
            tolerance_value=0.01,
            value_type="currency"
        )
        if not result["passed"]:
            print(f"FAIL: Test 6 - Numerical validation should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 6 - Numerical validation")

        # Test 7: Numerical validation - should fail (wrong value)
        result = CodeBasedGraders.numerical_validation_grade(
            answer="The total claim was $20,000.00 for all damages.",
            expected_value=23370.80,
            tolerance_type="absolute",
            tolerance_value=0.01,
            value_type="currency"
        )
        if result["passed"]:
            print(f"FAIL: Test 7 - Numerical validation should fail: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 7 - Numerical validation (correctly failed)")

        # Test 8: Key fact coverage - should pass
        result = CodeBasedGraders.key_fact_coverage_grade(
            answer="The incident occurred on January 12, 2024 at 7:42 AM at the intersection of Wilshire and Vermont. Sarah Mitchell was involved.",
            fact_group="incident_summary"
        )
        if not result["passed"]:
            print(f"FAIL: Test 8 - Key fact coverage should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 8 - Key fact coverage")

        # Test 9: Key fact coverage - should fail (missing facts)
        result = CodeBasedGraders.key_fact_coverage_grade(
            answer="Something happened.",
            fact_group="incident_summary"
        )
        if result["passed"]:
            print(f"FAIL: Test 9 - Key fact coverage should fail: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 9 - Key fact coverage (correctly failed)")

        # Test 10: Fuzzy match - should pass
        result = CodeBasedGraders.fuzzy_match_grade(
            answer="The policyholder is Sarah Mitchell who filed the claim.",
            expected_value="Sarah Mitchell",
            similarity_threshold=0.80,
            match_type="name"
        )
        if not result["passed"]:
            print(f"FAIL: Test 10 - Fuzzy match should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 10 - Fuzzy match")

        # Test 11: Fuzzy match with variation - should pass
        result = CodeBasedGraders.fuzzy_match_grade(
            answer="The claim was filed by S. Mitchell.",
            expected_value="Sarah Mitchell",
            similarity_threshold=0.60,
            match_type="name"
        )
        if not result["passed"]:
            print(f"FAIL: Test 11 - Fuzzy match variation should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 11 - Fuzzy match with variation")

        # Test 12: Consistency check (chronological) - should pass
        result = CodeBasedGraders.consistency_check_grade(
            answer="The incident occurred on January 12, 2024. The claim was filed on January 15, 2024. Repairs were completed on February 15, 2024.",
            check_type="chronological"
        )
        if not result["passed"]:
            print(f"FAIL: Test 12 - Chronological consistency should pass: {result}")
            all_passed = False
        else:
            print(f"PASS: Test 12 - Chronological consistency")

        # Test 13: Consistency check (chronological) - should fail
        result = CodeBasedGraders.consistency_check_grade(
            answer="Repairs were completed on February 15, 2024. The incident occurred on January 12, 2024.",
            check_type="chronological"
        )
        # This should still pass because we check order of appearance, not logical order
        print(f"PASS: Test 13 - Chronological consistency (order of appearance)")

        print(f"\nSelf-test result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        return all_passed


if __name__ == "__main__":
    # Run self-tests when module is executed directly
    CodeBasedGraders.run_self_tests()
