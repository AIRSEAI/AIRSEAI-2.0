"""Success criteria and failure classification, computed from state, not pixels.

Every criterion is a pure numpy function of a recorded `EpisodeTrace`, called
identically by the expert runner and the policy evaluator, and returns a
`Verdict` whose failure is one of the `FailureMode` taxonomy. No Isaac import.

Layering::

    criteria.trace        the recorded quantities and the verdict type
    criteria.detectors    task-independent failure tests
    criteria.probes       pose geometry both sides compute
    criteria.<task>       that task's probe gate and its `verdict`

A task module imports its own thresholds from `sim.dims.<task>` and nothing
from another task, so a task can be added or removed as whole files. This
package re-exports nothing: import each name from the module that defines it.
"""
