# Agent Roles

Agents are responsibilities, not magic. Each agent runs skills through the same
validation harness and must produce evidence before promotion.

| Agent | Owns | Cannot Do |
| --- | --- | --- |
| `orchestrator_agent` | skill order, config wiring, evidence bundle | bypass validators |
| `sim_agent` | Isaac Lab task checks, policy artifact audit, sim runs | approve real robot motion |
| `perception_agent` | shaft pose repeatability and calibration evidence | hide camera/calibration failures |
| `sysid_agent` | delay, stiction, action-scale, and actuator gap estimates | promote guesses without logs |
| `dr_agent` | bounded domain-randomization candidates | widen ranges without evidence |
| `autoresearch_agent` | hypotheses, experiment ranking, self-improvement loop | directly command hardware |
| `critic_agent` | regression comparison and evidence review | generate the candidate it reviews |
| `safety_agent` | release gate and human gate | set `safe_to_autorun_robot` true |

The release rule is simple: every release-blocking skill must pass, and real
robot execution remains human-gated.
