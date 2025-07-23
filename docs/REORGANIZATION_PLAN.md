# KIMERA SWM PROJECT REORGANIZATION PLAN
## Executed: 2025-01-08

### Current State Analysis
- **Total Python files**: 1,339
- **Archive directory**: 32MB of deprecated/broken code
- **Backend directory**: 8.6MB of production code
- **Structure**: Chaotic, doesn't follow KIMERA protocol

### Target Structure (KIMERA SWM Autonomous Architect Protocol)
```
/src/                           # Production-ready code only
├── core/                       # Invariant algorithms
├── models/                     # Neural architectures  
├── symbolic/                   # Symbolic processing
├── engines/                    # Processing engines
├── api/                        # API interfaces
├── security/                   # Security components
├── monitoring/                 # Health & performance monitoring
├── trading/                    # Trading system
├── pharmaceutical/             # Domain-specific modules
└── utils/                      # Shared utilities

/experiments/                   # All experimental work
└── {YYYY-MM-DD}_{name}/       # Isolated experiments
    ├── README.md               # Hypothesis & methodology
    ├── results/                # Outputs & metrics
    └── analysis.ipynb          # Interactive exploration

/tests/                         # Test suites
├── unit/                       # Mirrors src/ structure
├── integration/                # Cross-module tests
├── performance/                # Benchmark tests
└── adversarial/                # Failure-seeking tests

/archive/                       # Deprecated code
└── {YYYY-MM-DD}/              # Time-stamped archives
    └── DEPRECATED.md           # Explanation required

/docs/                          # Living documentation
├── architecture/               # System design
├── research/                   # Papers & notes
└── operations/                 # Runbooks

/config/                        # Configuration files
/data/                          # Data files
/cache/                         # Cache directory
/requirements/                  # Dependencies
/scripts/                       # Utility scripts
```

### Migration Steps

#### Phase 1: Create New Structure
1. Create `/src` directory with proper subdirectories
2. Create `/experiments` directory
3. Reorganize `/archive` with proper timestamping
4. Create `/tests` with proper subdirectories

#### Phase 2: Migrate Production Code
1. Move `/backend/core` → `/src/core`
2. Move `/backend/engines` → `/src/engines`  
3. Move `/backend/api` → `/src/api`
4. Move `/backend/security` → `/src/security`
5. Move `/backend/monitoring` → `/src/monitoring`
6. Move `/backend/trading` → `/src/trading`
7. Move `/backend/pharmaceutical` → `/src/pharmaceutical`
8. Move `/backend/utils` → `/src/utils`
9. Move `/backend/main.py` → `/src/main.py`

#### Phase 3: Archive Cleanup
1. Move `/archive/broken_scripts_and_tests` → `/archive/2025-01-08-legacy-cleanup`
2. Create proper `DEPRECATED.md` files
3. Consolidate duplicate archives

#### Phase 4: Tests & Documentation
1. Identify test files and move to `/tests`
2. Reorganize documentation in `/docs`
3. Update all import paths
4. Verify all `__init__.py` files

#### Phase 5: Verification
1. Run health check
2. Verify imports work
3. Update documentation
4. Generate migration report

### Risk Mitigation
- **No deletions**: Only moves and archives
- **Git tracking**: All moves tracked in git
- **Rollback plan**: Keep original structure until verification complete
- **Incremental**: Test each phase before proceeding

### Success Criteria
- [ ] All production code in `/src`
- [ ] All tests properly organized in `/tests`
- [ ] All experimental code in `/experiments`
- [ ] All deprecated code properly archived
- [ ] All imports working
- [ ] Documentation updated
- [ ] Clean project root 