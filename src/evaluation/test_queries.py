"""
Test Suite with Diverse Queries
5 Summary (General) queries + 5 Needle (Precise) queries
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSuite:
    """
    Comprehensive test suite for evaluating the insurance claim system
    5 Summary queries + 5 Needle queries = 10 total
    """

    @staticmethod
    def get_test_queries() -> List[Dict[str, Any]]:
        """
        Get all test queries with ground truth and expected behavior

        Returns:
            List of test query dictionaries
        """
        return [
            # ===== SUMMARY / GENERAL QUERIES (5) =====

            # Query 1: High-level summary
            {
                "id": "Q1_SUMMARY",
                "query": "What is this insurance claim about? Provide a summary.",
                "type": "summary",
                "expected_agent": "summarization",
                "expected_index": "summary",
                "ground_truth": "This is an auto insurance claim (CLM-2024-001) for a multi-vehicle collision that occurred on January 12, 2024, at 7:42 AM at the intersection of Wilshire Blvd and Vermont Ave in Los Angeles. Sarah Mitchell's 2021 Honda Accord was struck by Robert Harrison's vehicle which ran a red light while Harrison was driving under the influence (BAC 0.14). Mitchell sustained whiplash injuries, the vehicle required $17,111.83 in repairs, and the total claim amount was $23,370.80.",
                "expected_chunks": [
                    "Incident description",
                    "Timeline of events",
                    "Financial summary"
                ],
                "key_facts": [
                    "Multi-vehicle collision",
                    "DUI involved",
                    "January 12, 2024",
                    "Total claim ~$23,370"
                ]
            },

            # Query 2: Timeline query
            {
                "id": "Q2_TIMELINE",
                "query": "Provide a timeline of key events from the incident through vehicle return.",
                "type": "summary",
                "expected_agent": "summarization",
                "expected_index": "summary",
                "ground_truth": "January 12, 2024 (7:42 AM): Incident occurred. January 15, 2024: Claim filed. January 26, 2024: Liability accepted by at-fault party. January 29, 2024: Repairs authorized and commenced. February 15, 2024: Repairs completed. February 16, 2024: Vehicle returned to policyholder.",
                "expected_chunks": [
                    "Incident Timeline section",
                    "Post-incident timeline"
                ],
                "key_facts": [
                    "January 12 - incident",
                    "January 15 - filed",
                    "February 16 - vehicle returned"
                ]
            },

            # Query 3: Witness summary
            {
                "id": "Q3_WITNESSES",
                "query": "Who were the witnesses and what did they observe?",
                "type": "summary",
                "expected_agent": "summarization",
                "expected_index": "summary",
                "ground_truth": "There were three witnesses: Marcus Thompson (at bus stop) saw Harrison's Camry run a red light at high speed without braking. Elena Rodriguez (stopped in left turn lane) observed the Camry had a red light for several seconds before entering the intersection and noted Harrison appeared intoxicated after the crash. Patricia O'Brien (RN) confirmed the traffic signal timing and noted sunrise was at 6:58 AM with normal lighting conditions.",
                "expected_chunks": [
                    "Witness Statements section",
                    "All witness testimonies"
                ],
                "key_facts": [
                    "Marcus Thompson",
                    "Elena Rodriguez",
                    "Patricia O'Brien",
                    "ran red light"
                ]
            },

            # Query 4: Medical treatment summary
            {
                "id": "Q4_MEDICAL",
                "query": "Summarize the medical treatment Sarah Mitchell received.",
                "type": "summary",
                "expected_agent": "summarization",
                "expected_index": "summary",
                "ground_truth": "Sarah Mitchell was treated at Cedars-Sinai Emergency Department for cervical strain (whiplash) and post-traumatic headache. She had a follow-up with orthopedist Dr. Rachel Kim who prescribed physical therapy. She completed 8 physical therapy sessions at Pacific Coast Physical Therapy with Marcus Rodriguez, PT, DPT, from February 2-27, 2024.",
                "expected_chunks": [
                    "Medical Documentation section",
                    "Physical therapy records"
                ],
                "key_facts": [
                    "Cedars-Sinai ED",
                    "whiplash",
                    "Dr. Rachel Kim",
                    "8 PT sessions"
                ]
            },

            # Query 5: Liability outcome
            {
                "id": "Q5_LIABILITY",
                "query": "What was the outcome of the liability determination?",
                "type": "summary",
                "expected_agent": "summarization",
                "expected_index": "summary",
                "ground_truth": "Robert Harrison's insurance company (Nationwide Insurance) accepted 100% liability for the accident on January 26, 2024. Harrison was cited for DUI and running a red light. His BAC was 0.14%, above the legal limit of 0.08%.",
                "expected_chunks": [
                    "Liability section",
                    "Police Report section"
                ],
                "key_facts": [
                    "100% liability",
                    "January 26, 2024",
                    "DUI citation",
                    "0.14% BAC"
                ]
            },

            # ===== NEEDLE QUERIES (5) =====

            # Query 6: Deductible amount
            {
                "id": "Q6_NEEDLE_DEDUCTIBLE",
                "query": "What was the exact collision deductible amount?",
                "type": "needle",
                "expected_agent": "needle",
                "expected_index": "hierarchical",
                "ground_truth": "The collision deductible was exactly $750.",
                "expected_chunks": [
                    "Policy Information section",
                    "Deductible information"
                ],
                "key_facts": [
                    "$750",
                    "collision deductible"
                ]
            },

            # Query 7: Accident time
            {
                "id": "Q7_NEEDLE_TIME",
                "query": "At what exact time did the accident occur?",
                "type": "needle",
                "expected_agent": "needle",
                "expected_index": "hierarchical",
                "ground_truth": "The accident occurred at exactly 7:42 AM (more precisely 7:42:15 AM based on the incident timeline).",
                "expected_chunks": [
                    "Incident Timeline section"
                ],
                "key_facts": [
                    "7:42 AM",
                    "7:42:15 AM"
                ]
            },

            # Query 8: Claims adjuster
            {
                "id": "Q8_NEEDLE_ADJUSTER",
                "query": "Who was the claims adjuster assigned to this case?",
                "type": "needle",
                "expected_agent": "needle",
                "expected_index": "hierarchical",
                "ground_truth": "Kevin Park was the claims adjuster assigned to this case.",
                "expected_chunks": [
                    "Policy Information section"
                ],
                "key_facts": [
                    "Kevin Park",
                    "claims adjuster"
                ]
            },

            # Query 9: BAC level
            {
                "id": "Q9_NEEDLE_BAC",
                "query": "What was Robert Harrison's Blood Alcohol Concentration (BAC)?",
                "type": "needle",
                "expected_agent": "needle",
                "expected_index": "hierarchical",
                "ground_truth": "Robert Harrison's Blood Alcohol Concentration (BAC) was 0.14%, which is significantly above the legal limit of 0.08%.",
                "expected_chunks": [
                    "Police Report section"
                ],
                "key_facts": [
                    "0.14%",
                    "BAC",
                    "above legal limit"
                ]
            },

            # Query 10: PT sessions count
            {
                "id": "Q10_NEEDLE_PT",
                "query": "How many physical therapy sessions did Sarah Mitchell complete?",
                "type": "needle",
                "expected_agent": "needle",
                "expected_index": "hierarchical",
                "ground_truth": "Sarah Mitchell completed exactly 8 physical therapy sessions.",
                "expected_chunks": [
                    "Medical Documentation section",
                    "Physical therapy records"
                ],
                "key_facts": [
                    "8 sessions",
                    "physical therapy"
                ]
            }
        ]

    @staticmethod
    def get_query_by_id(query_id: str) -> Dict[str, Any]:
        """Get specific test query by ID"""
        queries = TestSuite.get_test_queries()
        for q in queries:
            if q["id"] == query_id:
                return q
        raise ValueError(f"Query ID {query_id} not found")

    @staticmethod
    def get_queries_by_type(query_type: str) -> List[Dict[str, Any]]:
        """Get all queries of a specific type"""
        queries = TestSuite.get_test_queries()
        return [q for q in queries if q["type"] == query_type]

    @staticmethod
    def print_summary():
        """Print summary of test suite"""
        queries = TestSuite.get_test_queries()

        print("\n" + "=" * 70)
        print("TEST SUITE SUMMARY")
        print("=" * 70)
        print(f"Total Queries: {len(queries)}\n")

        # Count by type
        types = {}
        for q in queries:
            qtype = q["type"]
            types[qtype] = types.get(qtype, 0) + 1

        print("Queries by Type:")
        for qtype, count in types.items():
            print(f"  {qtype}: {count}")

        print("\n" + "-" * 70)
        print("All Test Queries:")
        print("-" * 70)

        for i, q in enumerate(queries, 1):
            print(f"\n{i}. [{q['id']}] {q['type'].upper()}")
            print(f"   Query: {q['query']}")
            print(f"   Expected Agent: {q.get('expected_agent', 'N/A')}")
            print(f"   Expected Index: {q.get('expected_index', 'N/A')}")
            print(f"   Key Facts: {', '.join(q['key_facts'][:3])}...")


if __name__ == "__main__":
    TestSuite.print_summary()
