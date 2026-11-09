"""Small shape-only metric fixtures used by the E1--E9 unit tests."""

from benchmark_protocol_v140 import EXPECTED_LENGTH_COUNTS, EXPECTED_MODE_COUNTS, FORMAL_TEST_SUITES, SUITE_QUESTION_COUNTS


def formal_metric_fixture():
    result = {}
    for suite in FORMAL_TEST_SUITES:
        count = SUITE_QUESTION_COUNTS[suite]
        sides = {'prefix': count} if suite == 'test_prefix' else {'suffix': count} if suite == 'test_suffix' else {}
        result[suite] = {
            'questions': count,
            'by_length': {key: {'questions': value} for key, value in EXPECTED_LENGTH_COUNTS[suite].items()},
            'by_mode': {key: {'questions': value} for key, value in EXPECTED_MODE_COUNTS[suite].items()},
            'by_boundary_side': {key: {'questions': value} for key, value in sides.items()},
        }
    return result
