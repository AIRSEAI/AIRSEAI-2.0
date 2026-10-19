# SPDX-License-Identifier: Apache-2.0
"""Success criteria and failure classification, computed from recorded state.

Layering::

    criteria.trace        the recorded quantities and the verdict type
    criteria.detectors    task-independent failure tests
    criteria.probes       pose geometry both sides compute
    criteria.<task>       that task's probes and its `verdict`

Pure numpy, no Isaac. A task module imports only the shared modules and its
own `sim.dims.<task>`.
"""
