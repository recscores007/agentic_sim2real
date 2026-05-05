# Agent Roles

Agents are responsibilities, not magic. Each agent runs skills through the same
validation harness and must produce evidence before promotion.

The agent roles are intended to be portable. Robot-specific details should live
in config, adapters, real-data templates, and threshold policy, not in new
one-off agent roles.

| Agent | Owns | Cannot Do |
| --- | --- | --- |
| `orchestrator_agent` | skill order, config wiring, evidence bundle | bypass validators |
| `evaluator_agent` | real-data quality gates, canonical record checks, scored evidence | invent candidate parameters |
| `sim_agent` | Isaac Lab task checks, policy artifact audit, sim runs | approve real robot motion |
| `perception_agent` | object pose repeatability and calibration evidence | hide camera/calibration failures |
| `sysid_agent` | local and optional Newton SysID, delay, stiction, action-scale, and actuator gap estimates | promote guesses without logs |
| `dr_agent` | bounded domain-randomization candidates | widen ranges without evidence |
| `autoresearch_agent` | hypotheses, experiment ranking, self-improvement loop | directly command hardware |
| `critic_agent` | regression comparison and evidence review | generate the candidate it reviews |
| `safety_agent` | release gate and human gate | set `safe_to_autorun_robot` true |

The release rule is simple: every release-blocking skill must pass, and real
robot execution remains human-gated.
