"""
Expanded Test Suite with 40 Diverse Queries
Following Anthropic's best practices for comprehensive evaluation

20 Summary queries + 20 Needle queries = 40 total
Difficulty distribution: Easy (10), Medium (20), Hard (10)
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpandedTestSuite:
    """
    Comprehensive test suite with 40 queries
    20 Needle (precise facts) + 20 Summary (synthesis) queries
    """

    @staticmethod
    def get_all_queries() -> List[Dict[str, Any]]:
        """Get all 40 test queries"""
        return ExpandedTestSuite.get_needle_queries() + ExpandedTestSuite.get_summary_queries()

    @staticmethod
    def get_needle_queries() -> List[Dict[str, Any]]:
        """
        Get 20 needle queries for precise fact-finding
        Easy: 5, Medium: 10, Hard: 5
        """
        return [
            # ===== EASY NEEDLE QUERIES (5) =====

            {
                "id": "Q_NEEDLE_01",
                "query": "What is the claim ID?",
                "type": "needle",
                "difficulty": "easy",
                "category": "identifiers",
                "expected_agent": "needle",
                "ground_truth": "CLM-2024-001",
                "expected_facts": ["CLM-2024-001"],
                "expected_sections": ["CLAIM SUMMARY"],
                "graders": ["exact_match", "regex", "correctness"],
                "regex_pattern": r"CLM-\d{4}-\d{3}"
            },

            {
                "id": "Q_NEEDLE_02",
                "query": "Who is the policyholder?",
                "type": "needle",
                "difficulty": "easy",
                "category": "people",
                "expected_agent": "needle",
                "ground_truth": "Sarah Mitchell",
                "expected_facts": ["Sarah Mitchell"],
                "expected_sections": ["POLICY INFORMATION"],
                "graders": ["exact_match", "fuzzy_match", "correctness"]
            },

            {
                "id": "Q_NEEDLE_03",
                "query": "What was the collision deductible amount?",
                "type": "needle",
                "difficulty": "easy",
                "category": "financial",
                "expected_agent": "needle",
                "ground_truth": "$750",
                "expected_facts": ["$750", "collision deductible"],
                "expected_sections": ["POLICY INFORMATION"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            {
                "id": "Q_NEEDLE_04",
                "query": "When did the accident occur?",
                "type": "needle",
                "difficulty": "easy",
                "category": "dates",
                "expected_agent": "needle",
                "ground_truth": "January 12, 2024",
                "expected_facts": ["January 12, 2024"],
                "expected_sections": ["INCIDENT SUMMARY"],
                "graders": ["exact_match", "regex", "correctness"]
            },

            {
                "id": "Q_NEEDLE_05",
                "query": "How many physical therapy sessions did Sarah Mitchell complete?",
                "type": "needle",
                "difficulty": "easy",
                "category": "medical",
                "expected_agent": "needle",
                "ground_truth": "8",
                "expected_facts": ["8", "physical therapy"],
                "expected_sections": ["MEDICAL DOCUMENTATION"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            # ===== MEDIUM NEEDLE QUERIES (10) =====

            {
                "id": "Q_NEEDLE_06",
                "query": "What was Robert Harrison's BAC level?",
                "type": "needle",
                "difficulty": "medium",
                "category": "legal",
                "expected_agent": "needle",
                "ground_truth": "0.14%",
                "expected_facts": ["0.14%", "BAC", "Blood Alcohol Concentration"],
                "expected_sections": ["POLICE REPORT"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            {
                "id": "Q_NEEDLE_07",
                "query": "Who was the claims adjuster assigned to this case?",
                "type": "needle",
                "difficulty": "medium",
                "category": "people",
                "expected_agent": "needle",
                "ground_truth": "Kevin Park",
                "expected_facts": ["Kevin Park", "claims adjuster"],
                "expected_sections": ["POLICY INFORMATION"],
                "graders": ["exact_match", "fuzzy_match", "correctness"]
            },

            {
                "id": "Q_NEEDLE_08",
                "query": "What was the VIN of the insured vehicle?",
                "type": "needle",
                "difficulty": "medium",
                "category": "vehicle",
                "expected_agent": "needle",
                "ground_truth": "1HGCV1F39LA012345",
                "expected_facts": ["1HGCV1F39LA012345", "VIN"],
                "expected_sections": ["VEHICLE INFORMATION"],
                "graders": ["exact_match", "regex", "correctness"]
            },

            {
                "id": "Q_NEEDLE_09",
                "query": "At what exact time did the accident occur?",
                "type": "needle",
                "difficulty": "medium",
                "category": "dates",
                "expected_agent": "needle",
                "ground_truth": "7:42 AM (7:42:15 AM more precisely)",
                "expected_facts": ["7:42 AM", "7:42:15 AM"],
                "expected_sections": ["INCIDENT TIMELINE"],
                "graders": ["exact_match", "regex", "correctness"]
            },

            {
                "id": "Q_NEEDLE_10",
                "query": "What was the total claim amount?",
                "type": "needle",
                "difficulty": "medium",
                "category": "financial",
                "expected_agent": "needle",
                "ground_truth": "$23,370.80",
                "expected_facts": ["$23,370.80", "total claim"],
                "expected_sections": ["FINANCIAL SUMMARY"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            {
                "id": "Q_NEEDLE_11",
                "query": "What was the exact repair cost?",
                "type": "needle",
                "difficulty": "medium",
                "category": "financial",
                "expected_agent": "needle",
                "ground_truth": "$17,111.83",
                "expected_facts": ["$17,111.83", "repair cost"],
                "expected_sections": ["REPAIR DOCUMENTATION"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            {
                "id": "Q_NEEDLE_12",
                "query": "What is the at-fault driver's insurance company?",
                "type": "needle",
                "difficulty": "medium",
                "category": "insurance",
                "expected_agent": "needle",
                "ground_truth": "Nationwide Insurance",
                "expected_facts": ["Nationwide", "insurance"],
                "expected_sections": ["LIABILITY DETERMINATION"],
                "graders": ["exact_match", "fuzzy_match", "correctness"]
            },

            {
                "id": "Q_NEEDLE_13",
                "query": "Which hospital treated Sarah Mitchell?",
                "type": "needle",
                "difficulty": "medium",
                "category": "medical",
                "expected_agent": "needle",
                "ground_truth": "Cedars-Sinai Medical Center",
                "expected_facts": ["Cedars-Sinai", "hospital"],
                "expected_sections": ["MEDICAL DOCUMENTATION"],
                "graders": ["exact_match", "fuzzy_match", "correctness"]
            },

            {
                "id": "Q_NEEDLE_14",
                "query": "What was the make and model of the insured vehicle?",
                "type": "needle",
                "difficulty": "medium",
                "category": "vehicle",
                "expected_agent": "needle",
                "ground_truth": "2021 Honda Accord",
                "expected_facts": ["Honda Accord", "2021"],
                "expected_sections": ["VEHICLE INFORMATION"],
                "graders": ["multi_fact_extraction", "correctness"]
            },

            {
                "id": "Q_NEEDLE_15",
                "query": "What percentage liability was accepted?",
                "type": "needle",
                "difficulty": "medium",
                "category": "legal",
                "expected_agent": "needle",
                "ground_truth": "100%",
                "expected_facts": ["100%", "liability"],
                "expected_sections": ["LIABILITY DETERMINATION"],
                "graders": ["exact_match", "numerical_validation", "correctness"]
            },

            # ===== HARD NEEDLE QUERIES (5) =====

            {
                "id": "Q_NEEDLE_16",
                "query": "What traffic signal color did Elena Rodriguez witness for the at-fault driver?",
                "type": "needle",
                "difficulty": "hard",
                "category": "witness",
                "expected_agent": "needle",
                "ground_truth": "Red",
                "expected_facts": ["red", "traffic signal", "Elena Rodriguez"],
                "expected_sections": ["WITNESS STATEMENTS"],
                "graders": ["multi_fact_extraction", "cross_section_inference", "correctness"],
                "note": "Requires connecting witness to specific observation"
            },

            {
                "id": "Q_NEEDLE_17",
                "query": "How many seconds was the red light displayed before collision per Patricia O'Brien?",
                "type": "needle",
                "difficulty": "hard",
                "category": "witness",
                "expected_agent": "needle",
                "ground_truth": "Several seconds (exact count may not be specified)",
                "expected_facts": ["several seconds", "red light", "Patricia O'Brien"],
                "expected_sections": ["WITNESS STATEMENTS"],
                "graders": ["missing_information_detection", "query_understanding", "correctness"],
                "note": "Tests handling of potentially missing precise data"
            },

            {
                "id": "Q_NEEDLE_18",
                "query": "What was the sunrise time on the incident date?",
                "type": "needle",
                "difficulty": "hard",
                "category": "environmental",
                "expected_agent": "needle",
                "ground_truth": "6:58 AM",
                "expected_facts": ["6:58 AM", "sunrise"],
                "expected_sections": ["WITNESS STATEMENTS"],
                "graders": ["exact_match", "correctness"],
                "note": "Sparse fact in witness testimony"
            },

            {
                "id": "Q_NEEDLE_19",
                "query": "What was the adjuster's contact phone number?",
                "type": "needle",
                "difficulty": "hard",
                "category": "contact",
                "expected_agent": "needle",
                "ground_truth": "(213) 555-0147",
                "expected_facts": ["(213) 555-0147", "phone"],
                "expected_sections": ["POLICY INFORMATION"],
                "graders": ["exact_match", "regex", "correctness"],
                "note": "Tests disambiguation of multiple phone numbers"
            },

            {
                "id": "Q_NEEDLE_20",
                "query": "On what date was liability formally accepted?",
                "type": "needle",
                "difficulty": "hard",
                "category": "dates",
                "expected_agent": "needle",
                "ground_truth": "January 26, 2024",
                "expected_facts": ["January 26, 2024", "liability accepted"],
                "expected_sections": ["LIABILITY DETERMINATION", "TIMELINE"],
                "graders": ["exact_match", "date_arithmetic", "correctness"],
                "note": "Requires finding specific date in timeline"
            },
        ]

    @staticmethod
    def get_summary_queries() -> List[Dict[str, Any]]:
        """
        Get 20 summary queries for narrative synthesis
        Easy: 5, Medium: 10, Hard: 5
        """
        return [
            # ===== EASY SUMMARY QUERIES (5) =====

            {
                "id": "Q_SUMMARY_01",
                "query": "What is this insurance claim about? Provide a summary.",
                "type": "summary",
                "difficulty": "easy",
                "category": "overview",
                "expected_agent": "summarization",
                "ground_truth": "Auto insurance claim CLM-2024-001 for a multi-vehicle collision on January 12, 2024, at 7:42 AM. Sarah Mitchell's 2021 Honda Accord was struck by Robert Harrison who ran a red light while DUI (BAC 0.14%). Total claim: $23,370.80.",
                "expected_facts": ["CLM-2024-001", "January 12, 2024", "Sarah Mitchell", "Robert Harrison", "DUI", "$23,370.80"],
                "expected_sections": ["CLAIM SUMMARY", "INCIDENT SUMMARY"],
                "required_components": [
                    {"component": "claim_id", "weight": 1.0, "pattern": r"CLM-2024-001"},
                    {"component": "incident_date", "weight": 1.0, "pattern": r"January 12,? 2024"},
                    {"component": "parties", "weight": 2.0, "pattern": r"Sarah Mitchell.*Robert Harrison|Mitchell.*Harrison"},
                    {"component": "incident_type", "weight": 1.5, "pattern": r"collision|accident|crash"},
                    {"component": "dui", "weight": 1.5, "pattern": r"DUI|alcohol|intoxicated|BAC"},
                    {"component": "total_amount", "weight": 1.0, "pattern": r"\$23,370"}
                ],
                "graders": ["answer_completeness", "key_fact_coverage", "correctness"]
            },

            {
                "id": "Q_SUMMARY_02",
                "query": "Provide a timeline of key events from incident to vehicle return.",
                "type": "summary",
                "difficulty": "easy",
                "category": "timeline",
                "expected_agent": "summarization",
                "ground_truth": "Jan 12 (7:42 AM): Incident. Jan 15: Claim filed. Jan 26: Liability accepted. Jan 29: Repairs started. Feb 15: Repairs completed. Feb 16: Vehicle returned.",
                "expected_facts": ["January 12", "January 15", "January 26", "February 15", "February 16"],
                "expected_sections": ["TIMELINE"],
                "expected_events": [
                    {"event": "incident", "date": "January 12", "order": 1},
                    {"event": "claim_filed", "date": "January 15", "order": 2},
                    {"event": "liability_accepted", "date": "January 26", "order": 3},
                    {"event": "repairs_started", "date": "January 29", "order": 4},
                    {"event": "repairs_completed", "date": "February 15", "order": 5},
                    {"event": "vehicle_returned", "date": "February 16", "order": 6}
                ],
                "required_components": [
                    {"component": "incident", "weight": 1.5, "pattern": r"January 12|Jan.*12"},
                    {"component": "claim_filed", "weight": 1.0, "pattern": r"January 15|Jan.*15"},
                    {"component": "liability", "weight": 1.0, "pattern": r"January 26|Jan.*26"},
                    {"component": "repairs_completed", "weight": 1.0, "pattern": r"February 15|Feb.*15"},
                    {"component": "vehicle_returned", "weight": 1.5, "pattern": r"February 16|Feb.*16"}
                ],
                "graders": ["timeline_ordering", "answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_03",
                "query": "Who were the witnesses and what did they observe?",
                "type": "summary",
                "difficulty": "easy",
                "category": "witnesses",
                "expected_agent": "summarization",
                "ground_truth": "Marcus Thompson saw Harrison run red light at high speed. Elena Rodriguez confirmed red light and noted Harrison appeared intoxicated. Patricia O'Brien confirmed traffic signal timing and lighting conditions.",
                "expected_facts": ["Marcus Thompson", "Elena Rodriguez", "Patricia O'Brien", "red light"],
                "expected_sections": ["WITNESS STATEMENTS"],
                "required_facts": [
                    {"fact": "Marcus Thompson", "weight": 1.5, "pattern": r"Marcus Thompson"},
                    {"fact": "Elena Rodriguez", "weight": 1.5, "pattern": r"Elena Rodriguez"},
                    {"fact": "Patricia O'Brien", "weight": 1.0, "pattern": r"Patricia O'?Brien"},
                    {"fact": "red light", "weight": 2.0, "pattern": r"red light|ran.*light"}
                ],
                "required_information": [
                    {"info": "witness_names", "keywords": ["Thompson", "Rodriguez", "O'Brien"]},
                    {"info": "observations", "keywords": ["red light", "high speed", "intoxicated"]}
                ],
                "graders": ["multi_fact_extraction", "retrieval_coverage", "correctness"]
            },

            {
                "id": "Q_SUMMARY_04",
                "query": "Summarize the medical treatment Sarah Mitchell received.",
                "type": "summary",
                "difficulty": "easy",
                "category": "medical",
                "expected_agent": "summarization",
                "ground_truth": "Treated at Cedars-Sinai ED for whiplash and post-traumatic headache. Follow-up with Dr. Rachel Kim who prescribed PT. Completed 8 PT sessions Feb 2-27, 2024.",
                "expected_facts": ["Cedars-Sinai", "whiplash", "Dr. Rachel Kim", "8 sessions"],
                "expected_sections": ["MEDICAL DOCUMENTATION"],
                "required_components": [
                    {"component": "hospital", "weight": 1.0, "pattern": r"Cedars-Sinai|Emergency Department|ED"},
                    {"component": "diagnosis", "weight": 2.0, "pattern": r"whiplash|cervical strain"},
                    {"component": "doctor", "weight": 1.0, "pattern": r"Dr.*Rachel Kim|Rachel Kim"},
                    {"component": "pt_sessions", "weight": 1.5, "pattern": r"8.*session|eight.*session"}
                ],
                "graders": ["answer_completeness", "key_fact_coverage", "correctness"]
            },

            {
                "id": "Q_SUMMARY_05",
                "query": "What was the outcome of the liability determination?",
                "type": "summary",
                "difficulty": "easy",
                "category": "liability",
                "expected_agent": "summarization",
                "ground_truth": "Nationwide Insurance accepted 100% liability on January 26, 2024. Harrison cited for DUI (0.14% BAC) and running red light.",
                "expected_facts": ["100% liability", "January 26", "DUI", "0.14%"],
                "expected_sections": ["LIABILITY DETERMINATION"],
                "required_components": [
                    {"component": "liability_percentage", "weight": 2.0, "pattern": r"100%.*liabilit|full.*liabilit"},
                    {"component": "liability_date", "weight": 1.0, "pattern": r"January 26|Jan.*26"},
                    {"component": "dui", "weight": 1.5, "pattern": r"DUI|alcohol|0\.14"},
                    {"component": "red_light", "weight": 1.0, "pattern": r"red light|ran.*light"}
                ],
                "graders": ["answer_completeness", "correctness"]
            },

            # ===== MEDIUM SUMMARY QUERIES (10) =====

            {
                "id": "Q_SUMMARY_06",
                "query": "Describe the sequence of events leading to the collision.",
                "type": "summary",
                "difficulty": "medium",
                "category": "narrative",
                "expected_agent": "summarization",
                "ground_truth": "Harrison approached intersection at high speed without braking. Traffic signal had been red for several seconds. Harrison ran red light and struck Mitchell's vehicle in intersection. Impact occurred at 7:42:15 AM.",
                "expected_facts": ["high speed", "red light", "no braking", "intersection"],
                "expected_sections": ["INCIDENT DESCRIPTION", "WITNESS STATEMENTS"],
                "expected_events": [
                    {"event": "approaching", "description": "high speed", "order": 1},
                    {"event": "signal_red", "description": "red light", "order": 2},
                    {"event": "no_braking", "description": "no braking", "order": 3},
                    {"event": "collision", "description": "struck vehicle", "order": 4}
                ],
                "required_sections": ["INCIDENT DESCRIPTION", "WITNESS STATEMENTS"],
                "inference_check": {"required": "sequence of events", "evidence": "witness corroboration"},
                "graders": ["timeline_ordering", "cross_section_inference", "correctness"]
            },

            {
                "id": "Q_SUMMARY_07",
                "query": "What were all the costs involved in this claim and how do they add up?",
                "type": "summary",
                "difficulty": "medium",
                "category": "financial",
                "expected_agent": "summarization",
                "ground_truth": "Repair cost: $17,111.83. Medical costs and other expenses bring total to $23,370.80. Deductible: $750.",
                "expected_facts": ["$17,111.83", "$23,370.80", "$750"],
                "expected_sections": ["FINANCIAL SUMMARY", "REPAIR COSTS", "MEDICAL COSTS"],
                "financial_constraints": [
                    {"constraint": "repair_cost", "expected": 17111.83, "tolerance": 10},
                    {"constraint": "total_claim", "expected": 23370.80, "tolerance": 10},
                    {"constraint": "deductible", "expected": 750, "tolerance": 0}
                ],
                "required_components": [
                    {"component": "repair_cost", "weight": 2.0, "pattern": r"\$17,111"},
                    {"component": "total_cost", "weight": 2.0, "pattern": r"\$23,370"},
                    {"component": "deductible", "weight": 1.0, "pattern": r"\$750"}
                ],
                "graders": ["financial_constraint", "answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_08",
                "query": "Explain the medical treatment timeline from emergency care to PT completion.",
                "type": "summary",
                "difficulty": "medium",
                "category": "medical",
                "expected_agent": "summarization",
                "ground_truth": "Immediate: ED treatment at Cedars-Sinai for whiplash. Follow-up with Dr. Rachel Kim who prescribed PT. PT sessions from Feb 2-27, 2024 (8 total sessions).",
                "expected_facts": ["Cedars-Sinai", "Dr. Rachel Kim", "Feb 2-27", "8 sessions"],
                "expected_sections": ["MEDICAL DOCUMENTATION"],
                "expected_events": [
                    {"event": "emergency_dept", "date": "Jan 12", "order": 1},
                    {"event": "follow_up", "provider": "Dr. Kim", "order": 2},
                    {"event": "pt_start", "date": "Feb 2", "order": 3},
                    {"event": "pt_end", "date": "Feb 27", "order": 4}
                ],
                "required_components": [
                    {"component": "emergency", "weight": 1.5, "pattern": r"Cedars-Sinai|Emergency|ED"},
                    {"component": "doctor", "weight": 1.0, "pattern": r"Dr.*Kim|Rachel Kim"},
                    {"component": "pt_duration", "weight": 1.5, "pattern": r"Feb.*2.*27|8.*session"}
                ],
                "graders": ["timeline_ordering", "answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_09",
                "query": "Summarize the vehicle damage assessment findings.",
                "type": "summary",
                "difficulty": "medium",
                "category": "vehicle",
                "expected_agent": "summarization",
                "ground_truth": "Honda Accord sustained damage requiring $17,111.83 in repairs. Repairs completed between Jan 29 and Feb 15, 2024.",
                "expected_facts": ["Honda Accord", "$17,111.83", "repairs", "Jan 29", "Feb 15"],
                "expected_sections": ["VEHICLE DAMAGE", "REPAIR DOCUMENTATION"],
                "required_components": [
                    {"component": "vehicle", "weight": 1.0, "pattern": r"Honda Accord"},
                    {"component": "repair_cost", "weight": 2.0, "pattern": r"\$17,111"},
                    {"component": "repair_dates", "weight": 1.5, "pattern": r"Jan.*29.*Feb.*15|January.*29.*February.*15"}
                ],
                "graders": ["answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_10",
                "query": "What evidence supports the liability determination?",
                "type": "summary",
                "difficulty": "medium",
                "category": "legal",
                "expected_agent": "summarization",
                "ground_truth": "Multiple witnesses confirmed Harrison ran red light. Police cited Harrison for DUI (0.14% BAC) and running red light. Traffic signal timing confirmed by witness. Nationwide accepted 100% liability.",
                "expected_facts": ["witnesses", "red light", "DUI", "0.14%", "100% liability"],
                "expected_sections": ["WITNESS STATEMENTS", "POLICE REPORT", "LIABILITY"],
                "required_sections": ["WITNESS STATEMENTS", "POLICE REPORT", "LIABILITY"],
                "inference_check": {"required": "liability evidence", "evidence": "witnesses + police + insurance"},
                "required_components": [
                    {"component": "witnesses", "weight": 2.0, "pattern": r"witness"},
                    {"component": "police", "weight": 1.5, "pattern": r"police|DUI|cited"},
                    {"component": "bac", "weight": 1.0, "pattern": r"0\.14"},
                    {"component": "liability", "weight": 2.0, "pattern": r"100%|Nationwide.*accept"}
                ],
                "graders": ["cross_section_inference", "answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_11",
                "query": "Describe the claim processing timeline from filing to closure.",
                "type": "summary",
                "difficulty": "medium",
                "category": "process",
                "expected_agent": "summarization",
                "ground_truth": "Filed Jan 15. Liability accepted Jan 26. Repairs authorized and started Jan 29. Repairs completed Feb 15. Vehicle returned Feb 16.",
                "expected_facts": ["Jan 15", "Jan 26", "Jan 29", "Feb 15", "Feb 16"],
                "expected_sections": ["CLAIM TIMELINE"],
                "expected_events": [
                    {"event": "filed", "date": "Jan 15", "order": 1},
                    {"event": "liability", "date": "Jan 26", "order": 2},
                    {"event": "repairs_start", "date": "Jan 29", "order": 3},
                    {"event": "repairs_complete", "date": "Feb 15", "order": 4},
                    {"event": "vehicle_return", "date": "Feb 16", "order": 5}
                ],
                "required_components": [
                    {"component": "filing", "weight": 1.0, "pattern": r"Jan.*15"},
                    {"component": "liability", "weight": 1.5, "pattern": r"Jan.*26"},
                    {"component": "completion", "weight": 1.5, "pattern": r"Feb.*15|Feb.*16"}
                ],
                "graders": ["timeline_ordering", "answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_12",
                "query": "What policy coverage applied to this claim?",
                "type": "summary",
                "difficulty": "medium",
                "category": "policy",
                "expected_agent": "summarization",
                "ground_truth": "Policy POL-2024-VEH-45782 with collision coverage. Deductible: $750. Covered by at-fault party's liability insurance (Nationwide).",
                "expected_facts": ["POL-2024-VEH-45782", "collision coverage", "$750", "Nationwide"],
                "expected_sections": ["POLICY INFORMATION"],
                "required_components": [
                    {"component": "policy_number", "weight": 1.5, "pattern": r"POL-2024-VEH-45782"},
                    {"component": "coverage_type", "weight": 1.5, "pattern": r"collision"},
                    {"component": "deductible", "weight": 1.0, "pattern": r"\$750"},
                    {"component": "insurer", "weight": 1.0, "pattern": r"Nationwide"}
                ],
                "graders": ["answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_13",
                "query": "What actions did the claims adjuster take during processing?",
                "type": "summary",
                "difficulty": "medium",
                "category": "process",
                "expected_agent": "summarization",
                "ground_truth": "Kevin Park (adjuster) processed claim, coordinated with at-fault party's insurance, authorized repairs, managed timeline through vehicle return.",
                "expected_facts": ["Kevin Park", "claims adjuster", "coordinated", "authorized repairs"],
                "expected_sections": ["CLAIM PROCESSING"],
                "required_components": [
                    {"component": "adjuster_name", "weight": 1.5, "pattern": r"Kevin Park"},
                    {"component": "coordination", "weight": 1.5, "pattern": r"coordinat|at-fault.*insurance"},
                    {"component": "authorized", "weight": 1.5, "pattern": r"authoriz.*repair"},
                    {"component": "managed", "weight": 1.0, "pattern": r"manag|timeline|vehicle return"}
                ],
                "graders": ["answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_14",
                "query": "Describe the intersection where the accident occurred.",
                "type": "summary",
                "difficulty": "medium",
                "category": "location",
                "expected_agent": "summarization",
                "ground_truth": "Intersection of Wilshire Blvd and Vermont Ave in Los Angeles. Controlled by traffic signals. Sunrise at 6:58 AM, good lighting conditions at time of 7:42 AM incident.",
                "expected_facts": ["Wilshire", "Vermont", "Los Angeles", "traffic signals"],
                "expected_sections": ["INCIDENT LOCATION", "WITNESS STATEMENTS"],
                "required_components": [
                    {"component": "wilshire", "weight": 2.0, "pattern": r"Wilshire"},
                    {"component": "vermont", "weight": 2.0, "pattern": r"Vermont"},
                    {"component": "city", "weight": 1.0, "pattern": r"Los Angeles"},
                    {"component": "signals", "weight": 1.0, "pattern": r"traffic signal|signal"}
                ],
                "graders": ["answer_completeness", "correctness"]
            },

            {
                "id": "Q_SUMMARY_15",
                "query": "How long did the entire claim process take from incident to resolution?",
                "type": "summary",
                "difficulty": "medium",
                "category": "timeline",
                "expected_agent": "summarization",
                "ground_truth": "From Jan 12 incident to Feb 16 vehicle return: 35 days total. Claim filed within 3 days, liability accepted in 14 days, repairs took 17 days.",
                "expected_facts": ["Jan 12", "Feb 16", "35 days"],
                "expected_sections": ["TIMELINE"],
                "expected_calculation": {
                    "start_date": "January 12, 2024",
                    "end_date": "February 16, 2024",
                    "expected_days": 35
                },
                "required_components": [
                    {"component": "start_date", "weight": 1.0, "pattern": r"Jan.*12|January 12"},
                    {"component": "end_date", "weight": 1.0, "pattern": r"Feb.*16|February 16"},
                    {"component": "total_days", "weight": 2.0, "pattern": r"35.*day|thirty[- ]five.*day"}
                ],
                "graders": ["date_arithmetic", "answer_completeness", "correctness"]
            },

            # ===== HARD SUMMARY QUERIES (5) =====

            {
                "id": "Q_SUMMARY_16",
                "query": "Compare the witness statements - where do they agree and disagree?",
                "type": "summary",
                "difficulty": "hard",
                "category": "analysis",
                "expected_agent": "summarization",
                "ground_truth": "All three witnesses agree Harrison ran red light. Thompson and Rodriguez agree on high speed/no braking. O'Brien adds technical details on signal timing and lighting. No contradictions, statements corroborate each other.",
                "expected_facts": ["agree", "red light", "corroborate", "no contradictions"],
                "expected_sections": ["WITNESS STATEMENTS"],
                "required_sections": ["WITNESS STATEMENTS"],
                "inference_check": {"required": "comparative analysis", "evidence": "agreement and corroboration"},
                "graders": ["cross_section_inference", "coherence", "correctness"],
                "note": "Requires comparative analysis"
            },

            {
                "id": "Q_SUMMARY_17",
                "query": "Explain how the total claim amount was calculated and justified.",
                "type": "summary",
                "difficulty": "hard",
                "category": "financial",
                "expected_agent": "summarization",
                "ground_truth": "Repair costs ($17,111.83) + medical expenses + rental car + other costs = $23,370.80 total. Deductible of $750 applies. Costs justified by documented repairs, medical treatment, and verified expenses.",
                "expected_facts": ["$17,111.83", "$23,370.80", "$750", "justified"],
                "expected_sections": ["FINANCIAL SUMMARY", "COST BREAKDOWN"],
                "financial_constraints": [
                    {"constraint": "repair_cost", "expected": 17111.83, "tolerance": 10},
                    {"constraint": "total_claim", "expected": 23370.80, "tolerance": 10},
                    {"constraint": "deductible", "expected": 750, "tolerance": 0}
                ],
                "required_sections": ["FINANCIAL SUMMARY", "COST BREAKDOWN"],
                "inference_check": {"required": "cost calculation", "evidence": "itemized breakdown"},
                "graders": ["financial_constraint", "cross_section_inference", "correctness"],
                "note": "Requires understanding cost derivation"
            },

            {
                "id": "Q_SUMMARY_18",
                "query": "What factors contributed to the at-fault determination?",
                "type": "summary",
                "difficulty": "hard",
                "category": "legal",
                "expected_agent": "summarization",
                "ground_truth": "Multiple factors: (1) DUI (0.14% BAC), (2) Running red light confirmed by witnesses, (3) Excessive speed without braking, (4) Police citations, (5) Traffic signal timing. Combined evidence led to 100% liability.",
                "expected_facts": ["DUI", "red light", "witnesses", "citations", "100% liability"],
                "expected_sections": ["POLICE REPORT", "WITNESS STATEMENTS", "LIABILITY"],
                "required_facts": [
                    {"fact": "DUI/BAC", "weight": 2.0, "pattern": r"DUI|0\.14|alcohol"},
                    {"fact": "red_light", "weight": 2.0, "pattern": r"red light|ran.*light"},
                    {"fact": "witnesses", "weight": 1.5, "pattern": r"witness"},
                    {"fact": "citations", "weight": 1.5, "pattern": r"citat|police"},
                    {"fact": "liability", "weight": 2.0, "pattern": r"100%.*liabilit"}
                ],
                "required_sections": ["POLICE REPORT", "WITNESS STATEMENTS", "LIABILITY"],
                "inference_check": {"required": "multi-factor analysis", "evidence": "combined evidence"},
                "graders": ["multi_fact_extraction", "cross_section_inference", "correctness"],
                "note": "Requires multi-factor synthesis"
            },

            {
                "id": "Q_SUMMARY_19",
                "query": "Synthesize a coherent narrative of the incident from all available evidence.",
                "type": "summary",
                "difficulty": "hard",
                "category": "narrative",
                "expected_agent": "summarization",
                "ground_truth": "On Jan 12, 2024 at 7:42 AM, Robert Harrison (DUI, 0.14% BAC) approached Wilshire/Vermont intersection at high speed, ran red light without braking, and struck Sarah Mitchell's Honda Accord. Multiple witnesses confirmed red light violation. Mitchell sustained whiplash, received ER treatment and 8 PT sessions. Nationwide Insurance accepted 100% liability. Vehicle repairs ($17,111.83) completed, total claim $23,370.80.",
                "expected_facts": ["chronological flow", "all key parties", "outcome"],
                "expected_sections": ["ALL SECTIONS"],
                "expected_events": [
                    {"event": "incident", "date": "Jan 12", "time": "7:42 AM", "order": 1},
                    {"event": "collision", "description": "Harrison struck Mitchell", "order": 2},
                    {"event": "witnesses", "description": "confirmed violation", "order": 3},
                    {"event": "medical", "description": "ER and PT", "order": 4},
                    {"event": "liability", "description": "100% accepted", "order": 5},
                    {"event": "repairs", "cost": "$17,111.83", "order": 6},
                    {"event": "total", "cost": "$23,370.80", "order": 7}
                ],
                "required_components": [
                    {"component": "date_time", "weight": 1.5, "pattern": r"Jan.*12.*7:42|January 12.*7:42"},
                    {"component": "parties", "weight": 2.0, "pattern": r"Harrison.*Mitchell|Mitchell.*Harrison"},
                    {"component": "dui", "weight": 1.5, "pattern": r"DUI|0\.14"},
                    {"component": "location", "weight": 1.0, "pattern": r"Wilshire.*Vermont|intersection"},
                    {"component": "witnesses", "weight": 1.5, "pattern": r"witness"},
                    {"component": "injuries", "weight": 1.5, "pattern": r"whiplash|ER|PT"},
                    {"component": "liability", "weight": 1.5, "pattern": r"100%.*liabilit|Nationwide"},
                    {"component": "costs", "weight": 2.0, "pattern": r"\$17,111|\$23,370"}
                ],
                "graders": ["answer_completeness", "timeline_ordering", "coherence", "correctness"],
                "note": "Requires full document integration"
            },

            {
                "id": "Q_SUMMARY_20",
                "query": "If you were a lawyer representing Sarah Mitchell, what would be your strongest arguments?",
                "type": "summary",
                "difficulty": "hard",
                "category": "strategic",
                "expected_agent": "summarization",
                "ground_truth": "Strongest arguments: (1) Clear liability - DUI driver ran red light, (2) Multiple corroborating witnesses, (3) Police citations confirm violations, (4) Documented injuries and treatment, (5) Quantified damages with repair/medical costs, (6) At-fault insurance already accepted 100% liability.",
                "expected_facts": ["clear liability", "witnesses", "documented", "accepted liability"],
                "expected_sections": ["ALL SECTIONS"],
                "required_sections": ["POLICE REPORT", "WITNESS STATEMENTS", "MEDICAL DOCUMENTATION", "FINANCIAL SUMMARY", "LIABILITY"],
                "inference_check": {"required": "strategic legal arguments", "evidence": "multi-source corroboration"},
                "graders": ["cross_section_inference", "helpfulness", "correctness"],
                "note": "Requires strategic synthesis and reasoning"
            },
        ]

    @staticmethod
    def get_queries_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
        """Get queries filtered by difficulty"""
        all_queries = ExpandedTestSuite.get_all_queries()
        return [q for q in all_queries if q.get("difficulty") == difficulty]

    @staticmethod
    def get_queries_by_type(query_type: str) -> List[Dict[str, Any]]:
        """Get queries filtered by type"""
        if query_type == "needle":
            return ExpandedTestSuite.get_needle_queries()
        elif query_type == "summary":
            return ExpandedTestSuite.get_summary_queries()
        else:
            return []

    @staticmethod
    def print_summary():
        """Print summary of test suite"""
        all_queries = ExpandedTestSuite.get_all_queries()

        print("\n" + "=" * 80)
        print("EXPANDED TEST SUITE SUMMARY")
        print("=" * 80)
        print(f"Total Queries: {len(all_queries)}\n")

        # Count by type
        needle_count = len([q for q in all_queries if q["type"] == "needle"])
        summary_count = len([q for q in all_queries if q["type"] == "summary"])

        print(f"Needle Queries: {needle_count}")
        print(f"Summary Queries: {summary_count}\n")

        # Count by difficulty
        easy = len([q for q in all_queries if q.get("difficulty") == "easy"])
        medium = len([q for q in all_queries if q.get("difficulty") == "medium"])
        hard = len([q for q in all_queries if q.get("difficulty") == "hard"])

        print(f"Easy: {easy} ({easy/len(all_queries)*100:.0f}%)")
        print(f"Medium: {medium} ({medium/len(all_queries)*100:.0f}%)")
        print(f"Hard: {hard} ({hard/len(all_queries)*100:.0f}%)\n")

        # Category breakdown
        categories = {}
        for q in all_queries:
            cat = q.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    ExpandedTestSuite.print_summary()
