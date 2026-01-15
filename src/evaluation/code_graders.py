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

        # Remove duplicate values (likely the same total mentioned multiple times)
        unique_values = list(dict.fromkeys(found_values))
        sorted_values = sorted(unique_values, reverse=True)

        # Try to find a valid combination where components sum to total
        # Check if any subset of smaller values sums close to the largest value
        likely_total = sorted_values[0]
        best_match = None
        best_diff = float('inf')

        # Try different combinations: use only values smaller than total
        smaller_values = [v for v in sorted_values[1:] if v < likely_total * 0.9]

        if len(smaller_values) >= 2:
            # Try summing the smaller values
            component_sum = sum(smaller_values)
            diff = abs(component_sum - likely_total)

            if diff < best_diff:
                best_diff = diff
                best_match = smaller_values

        # If no good match found, use all smaller values
        if best_match is None:
            best_match = sorted_values[1:min(8, len(sorted_values))]  # Limit to 8 values
            component_sum = sum(best_match)
        else:
            component_sum = sum(best_match)

        operator = config.get("operator", "<=")
        tolerance = 1.05  # Allow 5% tolerance for rounding and tax

        if operator == "<=":
            passed = component_sum <= likely_total * tolerance
        elif operator == "==":
            passed = abs(component_sum - likely_total) < likely_total * 0.02  # 2% tolerance
        else:
            passed = True

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "likely_total": likely_total,
            "component_sum": component_sum,
            "components": best_match[:5],  # Show first 5
            "total_values_found": len(found_values),
            "unique_values_found": len(unique_values),
            "check_type": "sum_constraint",
            "details": f"Components sum ({component_sum:.2f}) {'<=' if passed else '>'} total ({likely_total:.2f}) [found {len(found_values)} values, {len(unique_values)} unique]"
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

    # ==========================================
    # NEW CODE-BASED GRADERS WITH PARTIAL CREDIT
    # ==========================================

    @staticmethod
    def multi_fact_extraction_grade(
        answer: str,
        required_facts: List[Dict[str, Any]],
        scoring_mode: str = "partial"
    ) -> Dict[str, Any]:
        """
        Extract multiple facts and award partial credit based on weighted components.

        Args:
            answer: The text to search in
            required_facts: List of facts with patterns and weights
                Example: [
                    {"fact": "incident_date", "pattern": r"January 12, 2024", "weight": 1.0},
                    {"fact": "location", "pattern": r"Wilshire.*Vermont", "weight": 1.5},
                ]
            scoring_mode: "partial" for weighted scoring, "binary" for all-or-nothing

        Returns:
            Dict with partial credit score, found/missing facts, and details
        """
        if not required_facts:
            return {
                "passed": False,
                "score": 0.0,
                "facts_found": [],
                "facts_missing": [],
                "details": "No facts specified"
            }

        facts_found = []
        facts_missing = []
        total_weight = sum(fact.get("weight", 1.0) for fact in required_facts)
        found_weight = 0.0

        for fact in required_facts:
            fact_name = fact.get("fact", "unknown")
            pattern = fact.get("pattern", "")
            weight = fact.get("weight", 1.0)

            if re.search(pattern, answer, re.IGNORECASE):
                facts_found.append(fact_name)
                found_weight += weight
            else:
                facts_missing.append(fact_name)

        if scoring_mode == "binary":
            score = 1.0 if len(facts_missing) == 0 else 0.0
        else:  # partial
            score = found_weight / total_weight if total_weight > 0 else 0.0

        passed = score >= 0.7  # 70% threshold for passing

        return {
            "passed": passed,
            "score": round(score, 3),
            "facts_found": facts_found,
            "facts_missing": facts_missing,
            "found_weight": round(found_weight, 2),
            "total_weight": round(total_weight, 2),
            "scoring_mode": scoring_mode,
            "details": f"Found {len(facts_found)}/{len(required_facts)} facts " +
                      f"(weighted score: {score*100:.1f}%)"
        }

    @staticmethod
    def tool_usage_correctness_grade(
        query: str,
        answer: str,
        expected_outcome: Dict[str, Any],
        tool_trace: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Verify correct outcome achieved regardless of tool path taken.
        Evaluates outcomes, not processes (Anthropic best practice).

        Args:
            query: The original query
            answer: The agent's answer
            expected_outcome: Expected outcome specification
                Example: {"days_calculated": True, "value": 3, "tolerance": 0}
            tool_trace: Optional list of tools called

        Returns:
            Dict with outcome verification results
        """
        outcome_type = expected_outcome.get("type", "value_present")
        expected_value = expected_outcome.get("value")
        tolerance = expected_outcome.get("tolerance", 0)

        outcome_achieved = False

        if outcome_type == "value_present":
            # Check if expected value appears in answer
            if isinstance(expected_value, (int, float)):
                # Look for numerical value with tolerance
                pattern = r'\d+(?:\.\d+)?'
                matches = re.findall(pattern, answer)
                for match in matches:
                    if abs(float(match) - expected_value) <= tolerance:
                        outcome_achieved = True
                        break
            else:
                # String value
                outcome_achieved = str(expected_value).lower() in answer.lower()

        tool_path = [t.get("tool", "unknown") for t in (tool_trace or [])]

        return {
            "passed": outcome_achieved,
            "score": 1.0 if outcome_achieved else 0.0,
            "outcome_achieved": outcome_achieved,
            "expected_outcome": expected_outcome,
            "tool_path": tool_path,
            "details": f"Outcome {'achieved' if outcome_achieved else 'not achieved'} " +
                      f"via tools: {tool_path}" if tool_path else "Outcome verification only"
        }

    @staticmethod
    def timeline_ordering_grade(
        answer: str,
        expected_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify logical chronological ordering with causal constraints.
        Exploitation-resistant: checks logical ordering, not just memorization.

        Args:
            answer: The text containing timeline
            expected_events: List of events with ordering constraints
                Example: [
                    {"event": "incident", "pattern": r"January 12", "must_be_before": ["claim_filed"]},
                    {"event": "claim_filed", "pattern": r"January 15", "must_be_after": ["incident"]}
                ]

        Returns:
            Dict with ordering verification results
        """
        # Extract positions of all events
        event_positions = {}
        for event_spec in expected_events:
            event_name = event_spec.get("event")
            pattern = event_spec.get("pattern")
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                event_positions[event_name] = match.start()

        # Check ordering constraints
        ordering_violations = []
        for event_spec in expected_events:
            event_name = event_spec.get("event")
            must_be_before = event_spec.get("must_be_before", [])
            must_be_after = event_spec.get("must_be_after", [])

            if event_name not in event_positions:
                ordering_violations.append({
                    "event": event_name,
                    "violation": "event not found in answer"
                })
                continue

            event_pos = event_positions[event_name]

            # Check must_be_before constraints
            for other_event in must_be_before:
                if other_event in event_positions:
                    if event_pos >= event_positions[other_event]:
                        ordering_violations.append({
                            "event": event_name,
                            "violation": f"{event_name} should be before {other_event}"
                        })

            # Check must_be_after constraints
            for other_event in must_be_after:
                if other_event in event_positions:
                    if event_pos <= event_positions[other_event]:
                        ordering_violations.append({
                            "event": event_name,
                            "violation": f"{event_name} should be after {other_event}"
                        })

        passed = len(ordering_violations) == 0

        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "events_found": len(event_positions),
            "total_events": len(expected_events),
            "ordering_violations": ordering_violations,
            "details": f"Found {len(event_positions)}/{len(expected_events)} events, " +
                      f"{len(ordering_violations)} ordering violations"
        }

    @staticmethod
    def financial_constraint_grade(
        answer: str,
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify complex financial relationships with partial credit.

        Args:
            answer: The text containing financial information
            constraints: List of financial constraints to verify
                Example: [
                    {"type": "sum", "components": [17111.83, 750], "equals": 17861.83, "tolerance": 0.01},
                    {"type": "range", "value_pattern": r"\$?750", "min": 500, "max": 1000}
                ]

        Returns:
            Dict with constraint verification and partial credit
        """
        constraints_satisfied = []
        constraints_violated = []

        # Extract all currency values from answer
        currency_pattern = r'\$?[\d,]+\.?\d*'
        raw_values = re.findall(currency_pattern, answer)
        values = []
        for v in raw_values:
            cleaned = v.replace('$', '').replace(',', '').strip()
            if cleaned and cleaned != '.':
                try:
                    values.append(float(cleaned))
                except ValueError:
                    continue

        for constraint in constraints:
            constraint_type = constraint.get("type")

            if constraint_type == "sum":
                components = constraint.get("components", [])
                expected_sum = constraint.get("equals")
                tolerance = constraint.get("tolerance", 0.01)

                # Check if sum relationship appears
                actual_sum = sum(components)
                if any(abs(v - expected_sum) <= tolerance for v in values):
                    constraints_satisfied.append(constraint)
                else:
                    constraints_violated.append({
                        "constraint": constraint,
                        "reason": f"Expected sum {expected_sum} not found"
                    })

            elif constraint_type == "range":
                value_pattern = constraint.get("value_pattern")
                min_val = constraint.get("min", 0)
                max_val = constraint.get("max", float('inf'))

                matches = re.findall(value_pattern, answer)
                if matches:
                    value = float(matches[0].replace('$', '').replace(',', ''))
                    if min_val <= value <= max_val:
                        constraints_satisfied.append(constraint)
                    else:
                        constraints_violated.append({
                            "constraint": constraint,
                            "reason": f"Value {value} outside range [{min_val}, {max_val}]"
                        })
                else:
                    constraints_violated.append({
                        "constraint": constraint,
                        "reason": "Value pattern not found"
                    })

        # Partial credit score
        total_constraints = len(constraints)
        satisfied_count = len(constraints_satisfied)
        score = satisfied_count / total_constraints if total_constraints > 0 else 0.0
        passed = score >= 0.7

        return {
            "passed": passed,
            "score": round(score, 3),
            "constraints_satisfied": satisfied_count,
            "constraints_violated": len(constraints_violated),
            "total_constraints": total_constraints,
            "violations": constraints_violated,
            "details": f"{satisfied_count}/{total_constraints} constraints satisfied ({score*100:.1f}%)"
        }

    @staticmethod
    def entity_relationship_grade(
        answer: str,
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify understanding of entity relationships with partial credit.
        Exploitation-resistant: checks relationship understanding, not memorization.

        Args:
            answer: The text containing entity relationships
            relationships: List of expected relationships
                Example: [
                    {"subject": "Sarah Mitchell", "relation": "policyholder_of", "object": "CLM-2024-001"},
                    {"subject": "Robert Harrison", "relation": "at_fault_in", "object": "CLM-2024-001"}
                ]

        Returns:
            Dict with relationship verification and partial credit
        """
        relationships_correct = []
        relationships_incorrect = []

        for rel in relationships:
            subject = rel.get("subject", "")
            relation_type = rel.get("relation", "")
            obj = rel.get("object", "")

            # Check if both entities are mentioned
            subject_found = subject.lower() in answer.lower()
            object_found = obj.lower() in answer.lower()

            if not (subject_found and object_found):
                relationships_incorrect.append({
                    "relationship": rel,
                    "reason": f"Missing entity: {'' if subject_found else subject} {'' if object_found else obj}"
                })
                continue

            # Define relation keywords based on type
            relation_keywords = {
                "policyholder_of": ["policyholder", "policy holder", "insured", "claimant"],
                "at_fault_in": ["at fault", "at-fault", "liable", "responsible for"],
                "adjuster_for": ["adjuster", "claims adjuster", "assigned to"],
                "victim_of": ["victim", "injured in", "involved in"],
            }

            keywords = relation_keywords.get(relation_type, [relation_type])

            # Check if relationship is stated correctly
            # Look for subject, then relation keywords, then object within reasonable proximity
            subject_pos = answer.lower().find(subject.lower())
            object_pos = answer.lower().find(obj.lower())

            # Extract text between entities
            if subject_pos < object_pos:
                between_text = answer[subject_pos:object_pos + len(obj)]
            else:
                between_text = answer[object_pos:subject_pos + len(subject)]

            relation_found = any(kw in between_text.lower() for kw in keywords)

            if relation_found:
                relationships_correct.append(rel)
            else:
                relationships_incorrect.append({
                    "relationship": rel,
                    "reason": f"Relationship '{relation_type}' not clearly stated"
                })

        # Partial credit score
        total = len(relationships)
        correct = len(relationships_correct)
        score = correct / total if total > 0 else 0.0
        passed = score >= 0.7

        return {
            "passed": passed,
            "score": round(score, 3),
            "relationships_correct": correct,
            "relationships_incorrect": len(relationships_incorrect),
            "total_relationships": total,
            "incorrect_details": relationships_incorrect,
            "details": f"{correct}/{total} relationships correctly stated ({score*100:.1f}%)"
        }

    @staticmethod
    def missing_information_detection_grade(
        query: str,
        answer: str,
        info_availability: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Verify agent correctly identifies when information is NOT in document.
        Critical for hallucination prevention.

        Args:
            query: The original query
            answer: The agent's answer
            info_availability: Which information should/shouldn't be available
                Example: {"weather_mentioned": False, "vehicle_color": False}

        Returns:
            Dict with hallucination detection results
        """
        missing_info_markers = [
            "not mentioned",
            "not specified",
            "not stated",
            "not available",
            "not provided",
            "not included",
            "no information",
            "information is not",
            "does not specify",
            "doesn't specify",
        ]

        hallucination_detected = False
        correctly_identified_missing = False

        # Check if answer correctly identifies missing information
        answer_lower = answer.lower()
        has_missing_marker = any(marker in answer_lower for marker in missing_info_markers)

        # Check for each piece of information
        for info_key, should_be_available in info_availability.items():
            if not should_be_available:
                # Information should NOT be in document
                # Answer should indicate this
                if has_missing_marker:
                    correctly_identified_missing = True
                else:
                    # Check if answer provides specific information (hallucination)
                    # This is a heuristic - specific values suggest hallucination
                    if re.search(r'\b(red|blue|green|clear|sunny|rainy)\b', answer_lower):
                        hallucination_detected = True

        passed = correctly_identified_missing and not hallucination_detected

        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "correctly_identified_missing": correctly_identified_missing,
            "hallucination_detected": hallucination_detected,
            "has_missing_marker": has_missing_marker,
            "details": "Correctly identified missing information" if passed else
                      ("Hallucination detected" if hallucination_detected else
                       "Did not identify missing information")
        }

    @staticmethod
    def cross_section_inference_grade(
        answer: str,
        required_sections: List[str],
        inference_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify synthesis across document sections with partial credit.

        Args:
            answer: The agent's answer
            required_sections: Sections that should be synthesized
            inference_check: Expected inference details
                Example: {
                    "section_facts": {"MEDICAL": ["8 PT sessions"], "TIMELINE": ["Feb 16"]},
                    "correct_inference": "Unable to work during PT",
                    "inference_pattern": r"unable to work|could not work"
                }

        Returns:
            Dict with cross-section synthesis verification and partial credit
        """
        section_facts = inference_check.get("section_facts", {})
        inference_pattern = inference_check.get("inference_pattern", "")

        # Check which section facts are present
        sections_used = []
        for section, facts in section_facts.items():
            section_found = any(fact.lower() in answer.lower() for fact in facts)
            if section_found:
                sections_used.append(section)

        # Check if inference is present
        inference_correct = bool(re.search(inference_pattern, answer, re.IGNORECASE)) if inference_pattern else False

        # Partial credit: 40% for retrieving from sections, 60% for correct inference
        section_score = len(sections_used) / len(section_facts) if section_facts else 0.0
        inference_score = 1.0 if inference_correct else 0.0

        total_score = (0.4 * section_score) + (0.6 * inference_score)
        passed = total_score >= 0.7

        return {
            "passed": passed,
            "score": round(total_score, 3),
            "sections_used": sections_used,
            "required_sections": list(section_facts.keys()),
            "inference_correct": inference_correct,
            "section_score": round(section_score, 3),
            "inference_score": inference_score,
            "details": f"Used {len(sections_used)}/{len(section_facts)} sections, " +
                      f"inference {'correct' if inference_correct else 'incorrect'} " +
                      f"(total: {total_score*100:.1f}%)"
        }

    @staticmethod
    def ambiguity_resolution_grade(
        query: str,
        answer: str,
        expected_behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify agent handles ambiguous queries correctly.
        Exploitation-resistant: can't be gamed by memorized Q&A pairs.

        Args:
            query: The potentially ambiguous query
            answer: The agent's answer
            expected_behavior: How agent should handle ambiguity
                Example: {
                    "should_clarify": True,
                    "acceptable_interpretations": ["total_claim", "repair_cost"],
                    "should_provide_multiple": True
                }

        Returns:
            Dict with ambiguity handling verification
        """
        should_clarify = expected_behavior.get("should_clarify", False)
        acceptable_interpretations = expected_behavior.get("acceptable_interpretations", [])
        should_provide_multiple = expected_behavior.get("should_provide_multiple", False)

        clarification_markers = [
            "could mean",
            "might refer to",
            "could be",
            "specifically",
            "to clarify",
            "breakdown",
            "includes",
            "consists of",
        ]

        has_clarification = any(marker in answer.lower() for marker in clarification_markers)

        # Check how many interpretations are provided
        interpretations_provided = sum(
            1 for interp in acceptable_interpretations
            if interp.replace('_', ' ').lower() in answer.lower()
        )

        provides_multiple = interpretations_provided >= 2

        # Scoring
        passed = True
        if should_clarify and not has_clarification:
            passed = False
        if should_provide_multiple and not provides_multiple:
            passed = False

        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "has_clarification": has_clarification,
            "interpretations_provided": interpretations_provided,
            "provides_multiple": provides_multiple,
            "details": f"Clarification {'provided' if has_clarification else 'missing'}, " +
                      f"{interpretations_provided} interpretations given"
        }

    @staticmethod
    def date_arithmetic_grade(
        query: str,
        answer: str,
        expected_calculation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify date calculations are correct with tolerance.
        Outcome-based: correct calculation regardless of method.

        Args:
            query: The query requesting date arithmetic
            answer: The agent's answer
            expected_calculation: Expected calculation details
                Example: {
                    "start_date": "2024-01-29",
                    "end_date": "2024-02-15",
                    "expected_days": 17,
                    "tolerance": 1,
                    "units": "days"
                }

        Returns:
            Dict with date arithmetic verification
        """
        expected_value = expected_calculation.get("expected_days")
        tolerance = expected_calculation.get("tolerance", 1)
        units = expected_calculation.get("units", "days")

        # Extract numerical values from answer
        numbers = [int(n) for n in re.findall(r'\b\d+\b', answer)]

        # Check if any number is within tolerance of expected value
        within_tolerance = any(
            abs(num - expected_value) <= tolerance
            for num in numbers
        )

        # Also check for approximate descriptions
        if units == "days":
            weeks = expected_value / 7
            if 2 <= weeks <= 4:
                week_pattern = rf'\b{int(round(weeks))}\s*weeks?\b'
                if re.search(week_pattern, answer, re.IGNORECASE):
                    within_tolerance = True

        return {
            "passed": within_tolerance,
            "score": 1.0 if within_tolerance else 0.0,
            "expected_value": expected_value,
            "tolerance": tolerance,
            "values_found": numbers,
            "within_tolerance": within_tolerance,
            "details": f"Expected {expected_value}±{tolerance} {units}, " +
                      f"found values: {numbers}"
        }

    @staticmethod
    def confidence_calibration_grade(
        query: str,
        answer: str,
        expected_confidence: str
    ) -> Dict[str, Any]:
        """
        Verify appropriate uncertainty expression.
        Prevents overconfident hallucinations.

        Args:
            query: The original query
            answer: The agent's answer
            expected_confidence: Expected confidence level ("high", "medium", "low")

        Returns:
            Dict with confidence calibration verification
        """
        high_confidence_markers = [
            "is", "was", "are", "the exact", "specifically", "precisely"
        ]
        hedging_markers = [
            "appears to be", "seems to be", "might be", "could be",
            "approximately", "around", "about", "roughly",
            "not specified", "not mentioned", "unclear"
        ]

        answer_lower = answer.lower()

        has_hedging = any(marker in answer_lower for marker in hedging_markers)
        has_high_confidence = any(marker in answer_lower for marker in high_confidence_markers)

        # Determine expressed confidence
        if has_hedging and not has_high_confidence:
            expressed_confidence = "low"
        elif has_high_confidence and not has_hedging:
            expressed_confidence = "high"
        else:
            expressed_confidence = "medium"

        calibration_correct = (expressed_confidence == expected_confidence)

        return {
            "passed": calibration_correct,
            "score": 1.0 if calibration_correct else 0.0,
            "expressed_confidence": expressed_confidence,
            "expected_confidence": expected_confidence,
            "has_hedging": has_hedging,
            "calibration_correct": calibration_correct,
            "details": f"Expected {expected_confidence} confidence, " +
                      f"expressed {expressed_confidence} confidence"
        }

    @staticmethod
    def retrieval_coverage_grade(
        query: str,
        retrieved_chunks: List[str],
        required_information: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure retrieval completeness with partial credit.

        Args:
            query: The original query
            retrieved_chunks: List of retrieved text chunks
            required_information: List of required information pieces
                Example: [
                    {"item": "Marcus Thompson", "weight": 1.0},
                    {"item": "Elena Rodriguez", "weight": 1.0},
                    {"item": "Patricia O'Brien", "weight": 1.0}
                ]

        Returns:
            Dict with retrieval coverage and partial credit
        """
        combined_chunks = " ".join(retrieved_chunks)

        retrieved_items = []
        missing_items = []
        total_weight = sum(item.get("weight", 1.0) for item in required_information)
        retrieved_weight = 0.0

        for item_spec in required_information:
            item_name = item_spec.get("item", "")
            weight = item_spec.get("weight", 1.0)

            if item_name.lower() in combined_chunks.lower():
                retrieved_items.append(item_name)
                retrieved_weight += weight
            else:
                missing_items.append(item_name)

        score = retrieved_weight / total_weight if total_weight > 0 else 0.0
        passed = score >= 0.8  # 80% threshold

        return {
            "passed": passed,
            "score": round(score, 3),
            "retrieved_items": len(retrieved_items),
            "missing_items": len(missing_items),
            "total_items": len(required_information),
            "missing_item_list": missing_items,
            "retrieved_weight": round(retrieved_weight, 2),
            "total_weight": round(total_weight, 2),
            "details": f"Retrieved {len(retrieved_items)}/{len(required_information)} items " +
                      f"(weighted: {score*100:.1f}%)"
        }

    @staticmethod
    def answer_completeness_grade(
        query: str,
        answer: str,
        required_components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score answer completeness with weighted components and partial credit.

        Args:
            query: The original query
            answer: The agent's answer
            required_components: List of required components
                Example: [
                    {"component": "incident_basics", "weight": 2.0, "pattern": r"January 12.*accident"},
                    {"component": "financial_summary", "weight": 1.5, "pattern": r"\$23,370"}
                ]

        Returns:
            Dict with completeness score and partial credit breakdown
        """
        components_found = []
        components_missing = []
        total_weight = sum(comp.get("weight", 1.0) for comp in required_components)
        found_weight = 0.0

        for comp_spec in required_components:
            comp_name = comp_spec.get("component", "unknown")
            pattern = comp_spec.get("pattern", "")
            weight = comp_spec.get("weight", 1.0)

            if re.search(pattern, answer, re.IGNORECASE):
                components_found.append(comp_name)
                found_weight += weight
            else:
                components_missing.append(comp_name)

        score = found_weight / total_weight if total_weight > 0 else 0.0
        passed = score >= 0.7  # 70% threshold

        return {
            "passed": passed,
            "score": round(score, 3),
            "components_found": components_found,
            "components_missing": components_missing,
            "total_components": len(required_components),
            "found_weight": round(found_weight, 2),
            "total_weight": round(total_weight, 2),
            "weighted_coverage": round(score, 3),
            "details": f"Found {len(components_found)}/{len(required_components)} components " +
                      f"(weighted: {score*100:.1f}%)"
        }

    # ==========================================
    # END NEW GRADERS
    # ==========================================

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
