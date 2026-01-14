"""
Test Explanations for Code-Based Graders

Provides detailed, user-friendly explanations for each test case.
"""

from typing import Dict

# Grader type explanations
GRADER_TYPE_EXPLANATIONS = {
    "exact_match": """
    **What it checks:** Verifies that specific expected values appear exactly in the RAG response.

    **Why it matters:** Ensures factual accuracy for critical information like IDs, names, and dates.

    **How it works:** Searches for the exact string (case-insensitive) within the response text.
    """,

    "regex": """
    **What it checks:** Tests if regex patterns correctly extract structured data from sample text.

    **Why it matters:** Validates that your extraction patterns work before using them in production.

    **How it works:** Applies regex pattern to predefined sample text and checks if expected values are found.
    """,

    "numerical_validation": """
    **What it checks:** Verifies numerical values (amounts, percentages, counts) are within acceptable tolerance.

    **Why it matters:** Catches calculation errors and ensures numerical accuracy while allowing for minor rounding differences.

    **How it works:** Extracts numbers from response, compares to expected value with configurable tolerance (absolute or percentage-based).
    """,

    "consistency_check": """
    **What it checks:** Validates internal consistency of facts within a response (chronological order, sum constraints, name consistency).

    **Why it matters:** Detects logical errors and contradictions that might confuse users even if individual facts are correct.

    **How it works:**
    - **Chronological**: Extracts dates and verifies they appear in chronological order
    - **Sum constraint**: Checks that component costs don't exceed total costs
    - **Name consistency**: Ensures a person's name is referenced consistently throughout
    """,

    "key_fact_coverage": """
    **What it checks:** Ensures responses contain all required facts for completeness.

    **Why it matters:** Prevents incomplete answers that miss critical information the user needs.

    **How it works:** Defines required facts per topic and checks if all are present in the response.
    """,

    "fuzzy_match": """
    **What it checks:** Validates names and strings allowing for minor variations (typos, abbreviations, word order).

    **Why it matters:** Real-world data often has variations - "Dr. Kim" vs "Dr. Rachel Kim" should both pass.

    **How it works:** Uses similarity scoring (Levenshtein distance) with configurable threshold (typically 80%+).
    """
}


# Test-specific explanations
TEST_EXPLANATIONS = {
    # Exact Match tests
    "CBG_RAG_01": {
        "title": "Claim ID Extraction",
        "what": "Checks if the claim ID 'CLM-2024-001' appears in the response.",
        "why": "The claim ID is the primary identifier - responses about claims must include it.",
        "pass": "Response contains 'CLM-2024-001'",
        "fail": "Response missing claim ID or has wrong ID"
    },
    "CBG_RAG_02": {
        "title": "Policyholder Name",
        "what": "Verifies 'Sarah Mitchell' is mentioned as the policyholder.",
        "why": "Accurately identifying the policyholder is critical for claim processing.",
        "pass": "Response contains 'Sarah Mitchell'",
        "fail": "Response missing or misspells policyholder name"
    },
    "CBG_RAG_03": {
        "title": "Collision Deductible Amount",
        "what": "Checks if '$750' deductible amount is stated.",
        "why": "Deductible amount affects financial calculations and claim settlement.",
        "pass": "Response mentions $750 deductible",
        "fail": "Response missing or has incorrect deductible amount"
    },
    "CBG_RAG_04": {
        "title": "Incident Date",
        "what": "Verifies the accident date 'January 12, 2024' is mentioned.",
        "why": "Date is essential for timeline reconstruction and claim validity.",
        "pass": "Response contains 'January 12, 2024'",
        "fail": "Response missing or has incorrect incident date"
    },
    "CBG_RAG_05": {
        "title": "Total Claim Amount",
        "what": "Checks if total claim amount '$23,370.80' appears.",
        "why": "Total claim amount is the most important financial figure.",
        "pass": "Response mentions $23,370.80",
        "fail": "Response missing or has incorrect total"
    },
    "CBG_RAG_06": {
        "title": "At-Fault Driver",
        "what": "Verifies 'Robert Harrison' is identified as at-fault driver.",
        "why": "Establishing fault determines liability and payment responsibility.",
        "pass": "Response contains 'Robert Harrison'",
        "fail": "Response missing or incorrect at-fault driver"
    },
    "CBG_RAG_07": {
        "title": "BAC Level",
        "what": "Checks if BAC of '0.14%' is mentioned.",
        "why": "BAC evidence is critical for DUI-related claims and liability.",
        "pass": "Response states 0.14% BAC",
        "fail": "Response missing or incorrect BAC level"
    },
    "CBG_RAG_08": {
        "title": "Claims Adjuster",
        "what": "Verifies 'Kevin Park' is mentioned as the adjuster.",
        "why": "Adjuster is the primary contact for claim status and questions.",
        "pass": "Response contains 'Kevin Park'",
        "fail": "Response missing or incorrect adjuster name"
    },
    "CBG_RAG_09": {
        "title": "Physical Therapy Sessions",
        "what": "Checks if '8' PT sessions is stated.",
        "why": "Medical treatment counts affect injury claim calculations.",
        "pass": "Response mentions 8 sessions",
        "fail": "Response missing or incorrect session count"
    },
    "CBG_RAG_10": {
        "title": "Vehicle Repair Cost",
        "what": "Verifies repair cost of '$17,111.83'.",
        "why": "Repair cost is a major component of the total claim amount.",
        "pass": "Response states $17,111.83 repair cost",
        "fail": "Response missing or incorrect repair cost"
    },

    # Numerical validation tests
    "CBG_NUM_01": {
        "title": "Total Claim Amount (Numerical)",
        "what": "Validates total claim of $23,370.80 with ±$0.01 tolerance.",
        "why": "Allows for minor rounding differences while ensuring accuracy.",
        "pass": "Extracted number within $0.01 of expected",
        "fail": "Number outside tolerance range"
    },
    "CBG_NUM_02": {
        "title": "Repair Cost (Percentage Tolerance)",
        "what": "Validates repair cost with 1% tolerance (~$171).",
        "why": "Allows for estimate variations while catching major errors.",
        "pass": "Within 1% of $17,111.83",
        "fail": "Differs by more than 1%"
    },
    "CBG_NUM_03": {
        "title": "BAC Percentage",
        "what": "Validates BAC of 0.14 with ±0.001 tolerance.",
        "why": "BAC must be precise as it has legal implications.",
        "pass": "0.139 to 0.141",
        "fail": "Outside narrow tolerance range"
    },
    "CBG_NUM_04": {
        "title": "Deductible Amount",
        "what": "Validates $750 deductible with ±$0.01 tolerance.",
        "why": "Deductible is a fixed contractual amount.",
        "pass": "$749.99 to $750.01",
        "fail": "Outside tolerance"
    },
    "CBG_NUM_05": {
        "title": "PT Session Count",
        "what": "Validates exactly 8 sessions (no tolerance).",
        "why": "Session counts are discrete integers, must be exact.",
        "pass": "Exactly 8",
        "fail": "Any other number"
    },

    # Consistency checks
    "CBG_CONS_01": {
        "title": "Chronological Order",
        "what": "Verifies dates appear in chronological order as you read the text.",
        "why": "Out-of-order dates confuse readers and suggest disorganized thinking.",
        "pass": "All dates in ascending chronological order",
        "fail": "A later date appears before an earlier date in the text"
    },
    "CBG_CONS_02": {
        "title": "Sum Constraint",
        "what": "Checks that component costs don't exceed total costs.",
        "why": "Component costs summing to more than total is mathematically impossible.",
        "pass": "Sum of components ≤ total (with 5% tolerance for tax/rounding)",
        "fail": "Components exceed total by >5%"
    },
    "CBG_CONS_03": {
        "title": "Name Consistency",
        "what": "Ensures 'Sarah Mitchell' is referenced consistently (not as 'Sarah', 'Mitchell', or variations).",
        "why": "Inconsistent naming confuses readers about whether it's the same person.",
        "pass": "All references use consistent form",
        "fail": "Multiple different name variations found"
    },

    # Coverage checks
    "CBG_FACT_01": {
        "title": "Incident Summary Coverage",
        "what": "Checks if response includes: date, location, vehicles involved, at-fault driver.",
        "why": "A complete incident summary must cover all essential facts.",
        "pass": "All required facts present",
        "fail": "Missing one or more required facts"
    },
    "CBG_FACT_02": {
        "title": "Financial Summary Coverage",
        "what": "Checks if response includes: total claim, repair cost, medical costs, deductible.",
        "why": "Financial summaries must show complete picture of costs.",
        "pass": "All cost components mentioned",
        "fail": "Missing financial information"
    },
    "CBG_FACT_03": {
        "title": "Liability Determination Coverage",
        "what": "Checks if response includes: fault determination, evidence, BAC level, police report.",
        "why": "Liability decisions must be backed by complete evidence.",
        "pass": "All liability factors covered",
        "fail": "Incomplete liability information"
    },
    "CBG_FACT_04": {
        "title": "Medical Treatment Coverage",
        "what": "Checks if response includes: hospital, doctor, diagnosis, treatment plan.",
        "why": "Medical summaries should cover all aspects of treatment.",
        "pass": "All medical facts present",
        "fail": "Missing medical details"
    },
    "CBG_FACT_05": {
        "title": "Witness Information Coverage",
        "what": "Checks if response includes witness names and their observations.",
        "why": "Witness accounts must identify who saw what.",
        "pass": "All witnesses and observations mentioned",
        "fail": "Missing witness names or observations"
    },

    # Fuzzy match tests
    "CBG_FUZZY_01": {
        "title": "Policyholder Name (Fuzzy)",
        "what": "Matches 'Sarah Mitchell' with 80% similarity threshold.",
        "why": "Handles variations like 'S. Mitchell', 'Sarah J. Mitchell', minor typos.",
        "pass": "≥80% similar to 'Sarah Mitchell'",
        "fail": "<80% similar or completely different name"
    },
    "CBG_FUZZY_02": {
        "title": "At-Fault Driver (Fuzzy)",
        "what": "Matches 'Robert Harrison' with 80% similarity.",
        "why": "Handles variations like 'R. Harrison', 'Rob Harrison'.",
        "pass": "≥80% similar to 'Robert Harrison'",
        "fail": "<80% similar"
    },
    "CBG_FUZZY_03": {
        "title": "Claims Adjuster (Fuzzy)",
        "what": "Matches 'Kevin Park' with 75% similarity (lower threshold).",
        "why": "Adjuster names might be abbreviated or formatted differently.",
        "pass": "≥75% similar to 'Kevin Park'",
        "fail": "<75% similar"
    },
    "CBG_FUZZY_04": {
        "title": "Hospital Name (Fuzzy)",
        "what": "Matches 'Cedars-Sinai' with 85% similarity.",
        "why": "Hospital names might have spacing/hyphen variations.",
        "pass": "≥85% similar to 'Cedars-Sinai'",
        "fail": "<85% similar"
    },
    "CBG_FUZZY_05": {
        "title": "Doctor Name (Fuzzy)",
        "what": "Matches 'Dr. Rachel Kim' with 70% similarity.",
        "why": "Doctor names might be formatted as 'Rachel Kim, M.D.' or 'Dr. R. Kim'.",
        "pass": "≥70% similar to 'Dr. Rachel Kim'",
        "fail": "<70% similar"
    },

    # Regex validation tests
    "CBG_REGEX_01": {
        "title": "Claim ID Pattern",
        "what": "Tests regex pattern: CLM-YYYY-NNN (e.g., CLM-2024-001).",
        "why": "Validates pattern can extract claim IDs from text.",
        "pass": "Pattern matches expected claim IDs",
        "fail": "Pattern doesn't match or matches incorrectly"
    },
    "CBG_REGEX_02": {
        "title": "Currency Pattern",
        "what": "Tests regex for $X,XXX.XX format.",
        "why": "Validates currency extraction with commas and cents.",
        "pass": "Matches currency amounts correctly",
        "fail": "Misses amounts or matches incorrectly"
    },
    "CBG_REGEX_03": {
        "title": "Date Pattern",
        "what": "Tests regex for 'Month DD, YYYY' format.",
        "why": "Validates date extraction in common format.",
        "pass": "Matches dates like 'January 12, 2024'",
        "fail": "Misses dates or matches incorrectly"
    },
    "CBG_REGEX_04": {
        "title": "Time Pattern",
        "what": "Tests regex for 'HH:MM AM/PM' format.",
        "why": "Validates time extraction from text.",
        "pass": "Matches times like '7:42 AM'",
        "fail": "Misses times or matches incorrectly"
    },
    "CBG_REGEX_05": {
        "title": "VIN Pattern",
        "what": "Tests regex for 17-character VIN.",
        "why": "Validates vehicle identification number extraction.",
        "pass": "Matches 17-char VINs",
        "fail": "Misses VINs or wrong length"
    },
    "CBG_REGEX_06": {
        "title": "Phone Pattern",
        "what": "Tests regex for (XXX) XXX-XXXX format.",
        "why": "Validates phone number extraction.",
        "pass": "Matches phone numbers correctly",
        "fail": "Misses numbers or matches incorrectly"
    },
    "CBG_REGEX_07": {
        "title": "Percentage Pattern",
        "what": "Tests regex for X.XX% format.",
        "why": "Validates percentage extraction (like BAC).",
        "pass": "Matches percentages like '0.14%'",
        "fail": "Misses percentages or matches incorrectly"
    },
    "CBG_REGEX_08": {
        "title": "Policy Number Pattern",
        "what": "Tests regex for POL-YYYY-XXX-NNNNN format.",
        "why": "Validates policy number extraction.",
        "pass": "Matches policy numbers correctly",
        "fail": "Misses policy numbers or matches incorrectly"
    },
}


def get_test_explanation(test_id: str) -> Dict[str, str]:
    """Get detailed explanation for a specific test."""
    return TEST_EXPLANATIONS.get(test_id, {
        "title": "Test Explanation",
        "what": "No detailed explanation available for this test.",
        "why": "Check test case definition for more information.",
        "pass": "Test criteria met",
        "fail": "Test criteria not met"
    })


def get_grader_type_explanation(grader_type: str) -> str:
    """Get explanation for a grader type."""
    return GRADER_TYPE_EXPLANATIONS.get(grader_type, "No explanation available for this grader type.")


def format_test_explanation(test_id: str, result: Dict) -> str:
    """
    Format a test explanation with result details.

    Args:
        test_id: Test case ID
        result: Test result dictionary

    Returns:
        Formatted markdown string
    """
    exp = get_test_explanation(test_id)
    status = "✅ PASSED" if result.get("passed") else "❌ FAILED"

    explanation = f"""
### {exp['title']} - {status}

**What this test checks:**
{exp['what']}

**Why it matters:**
{exp['why']}

**Pass criteria:** {exp['pass']}
**Fail criteria:** {exp['fail']}

**Result:** {result.get('details', 'No details available')}
"""

    return explanation
