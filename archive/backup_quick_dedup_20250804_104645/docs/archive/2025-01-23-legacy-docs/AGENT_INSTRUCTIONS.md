# KIMERA SWM AGENT INSTRUCTIONS
## Operational Procedures for Complex Multi-File Operations

---

## SECTION VII: OPERATIONAL PROCEDURES FOR AGENT MODE

### 7.1 Session Initialization
```markdown
On each session start:
1. Run health check: `python scripts/health_check.py`
2. Review outstanding issues in CURRENT_ISSUES.md
3. Check experiment status in experiments/
4. Verify test suite passes
5. Report any anomalies before proceeding
```

### 7.2 Daily Workflow Procedures

#### Morning: Research Planning
```markdown
1. Review hypotheses in experiments/active/
2. Design today's experiments with:
   - Clear hypothesis
   - Control variables
   - Success metrics
   - Failure analysis
3. Identify cross-domain inspiration opportunities
```

#### Development: Creative Implementation
```markdown
1. State approach before coding
2. Build simplest version first
3. Test edge cases immediately
4. Try "impossible" alternatives
5. Document surprises and insights
```

#### Evening: Integration & Learning
```markdown
1. Run full test suite
2. Update documentation
3. Archive failed experiments with lessons
4. Promote successful patterns to src/
5. Update LESSONS_LEARNED.md
```

---

## COMPLEX MULTI-FILE OPERATION PROCEDURES

### 1. Large-Scale Refactoring Protocol

When refactoring across multiple files:

```yaml
pre_refactoring:
  - Create comprehensive dependency graph
  - Identify all affected modules
  - Generate test coverage report
  - Create rollback branch
  
execution:
  - Work in atomic, testable chunks
  - Run tests after each chunk
  - Update imports incrementally
  - Maintain backwards compatibility
  
post_refactoring:
  - Verify all tests pass
  - Check performance metrics
  - Update documentation
  - Create migration guide
```

### 2. Cross-Module Integration Protocol

For integrating new features across modules:

```yaml
planning:
  - Map integration touchpoints
  - Design interface contracts
  - Plan data flow paths
  - Identify potential conflicts

implementation:
  - Start with interface definitions
  - Implement mock versions first
  - Integrate incrementally
  - Test at each integration point

validation:
  - End-to-end integration tests
  - Performance benchmarking
  - Security audit
  - Documentation update
```

### 3. Architecture Evolution Protocol

When evolving system architecture:

```yaml
analysis:
  - Document current architecture
  - Identify pain points
  - Research alternative patterns
  - Create evolution roadmap

prototyping:
  - Build proof-of-concept
  - Test with real workloads
  - Measure improvements
  - Document learnings

migration:
  - Create parallel implementation
  - Gradual traffic shifting
  - Monitor key metrics
  - Maintain rollback capability
```

### 4. Emergency Response Protocol

For critical system issues:

```yaml
immediate_response:
  - Assess severity and scope
  - Implement temporary mitigation
  - Alert relevant stakeholders
  - Begin root cause analysis

investigation:
  - Reproduce issue in isolation
  - Analyze logs and metrics
  - Test potential fixes
  - Document findings

resolution:
  - Implement permanent fix
  - Add regression tests
  - Update monitoring
  - Post-mortem analysis
```

### 5. Scientific Experiment Protocol

For research experiments:

```yaml
design:
  - Formulate clear hypothesis
  - Define success metrics
  - Plan control experiments
  - Prepare data collection

execution:
  - Set random seeds
  - Log all parameters
  - Capture intermediate results
  - Monitor resource usage

analysis:
  - Statistical validation
  - Visualization creation
  - Peer review preparation
  - Results documentation
```

---

## AUTOMATION SCRIPTS

### Health Check Automation
```bash
#!/bin/bash
# scripts/daily_health_check.sh

echo "=== Kimera SWM Daily Health Check ==="
date

# Check Python environment
python --version
pip check

# Run test suite
pytest -v --tb=short

# Check code quality
black --check backend/
mypy backend/ --ignore-missing-imports

# System metrics
python scripts/system_metrics.py

# Generate report
python scripts/generate_health_report.py
```

### Experiment Runner
```python
# scripts/run_experiment.py
import json
import datetime
from pathlib import Path

def run_experiment(name, hypothesis, function):
    """Run and document an experiment"""
    exp_dir = Path(f"experiments/{datetime.date.today()}_{name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "name": name,
        "hypothesis": hypothesis,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    
    # Run experiment
    results = function()
    
    # Save results
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results
```

---

## CONTINUOUS IMPROVEMENT

### Weekly Review Checklist
- [ ] Review all experiments from the week
- [ ] Identify successful patterns
- [ ] Document lessons learned
- [ ] Update best practices
- [ ] Plan next week's experiments

### Monthly Architecture Review
- [ ] Analyze system metrics trends
- [ ] Review technical debt
- [ ] Update dependency versions
- [ ] Plan refactoring priorities
- [ ] Update documentation

### Quarterly Strategic Assessment
- [ ] Evaluate research progress
- [ ] Review architecture decisions
- [ ] Update development roadmap
- [ ] Assess tool effectiveness
- [ ] Plan protocol updates

---

## REMEMBER

1. **Always verify before proceeding** - Trust nothing, verify everything
2. **Document as you go** - Future you will thank present you
3. **Test incrementally** - Small steps prevent big falls
4. **Embrace failure** - Every error teaches something valuable
5. **Stay curious** - The best solutions come from unexpected places

*Protocol Version: 3.0 | Last Updated: 2025-01-23* 