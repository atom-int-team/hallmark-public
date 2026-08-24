"""hallmark — universal materials-design benchmark.

Three separated concerns (see UNIVERSAL_BENCHMARK_SPEC §4):
  1. Task + grading spec   (system-agnostic YAML)   -> hallmark.loader / hallmark.models
  2. SUT adapter -> Transcript (cached responses)    -> hallmark.adapters
  3. Capability-aware graders -> Result              -> hallmark.graders / hallmark.scoring
"""

__version__ = "0.1.0"
