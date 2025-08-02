# KIMERA SWM Dependency Consolidation Report
**Generated**: 2025-07-31T22:28:35.105807
**Total Packages**: 584
**Conflicts Found**: 107

## Conflict Resolution Summary

### fastapi
**Conflicting versions:**

- `>=0.104.0` from: api.txt
- `==0.115.13` from: base.txt, requirements.txt
- `>=0.115.0          # API framework` from: thermodynamic.txt

---

### pydantic
**Conflicting versions:**

- `>=2.5.0` from: api.txt
- `==2.8.2 # Newer versions available, using one from a similar project` from: base.txt
- `==2.8.2` from: requirements.txt
- `>=2.8.0           # Data validation and settings` from: thermodynamic.txt

---

### python-multipart
**Conflicting versions:**

- `>=0.0.6` from: api.txt, api.txt
- `==0.0.20` from: requirements.txt

---

### httpx
**Conflicting versions:**

- `>=0.25.0` from: api.txt
- `==0.28.1` from: base.txt, requirements.txt

---

### requests
**Conflicting versions:**

- `>=2.31.0` from: api.txt
- `==2.32.3` from: base.txt, requirements.txt

---

### redis
**Conflicting versions:**

- `>=5.0.0` from: api.txt
- `==5.0.1` from: omnidimensional.txt
- `==6.2.0` from: requirements.txt

---

### asyncpg
**Conflicting versions:**

- `>=0.29.0` from: api.txt
- `==0.30.0` from: requirements.txt

---

### sqlalchemy
**Conflicting versions:**

- `>=2.0.0` from: api.txt
- `==2.0.31 # From your provided list, but newer versions exist` from: base.txt
- `==2.0.31` from: requirements.txt

---

### alembic
**Conflicting versions:**

- `>=1.12.0` from: api.txt
- `==1.16.2` from: base.txt, requirements.txt

---

### prometheus-client
**Conflicting versions:**

- `>=0.19.0` from: api.txt
- `==0.22.1` from: requirements.txt

---

### pytest
**Conflicting versions:**

- `>=7.4.0` from: api.txt
- `` from: dev.txt, testing.txt
- `==8.4.1` from: requirements.txt
- `>=7.4.0            # Testing framework` from: thermodynamic.txt

---

### pytest-asyncio
**Conflicting versions:**

- `>=0.21.0` from: api.txt
- `` from: dev.txt, testing.txt
- `==0.23.2  # For async testing` from: omnidimensional.txt
- `==1.0.0` from: requirements.txt
- `>=0.21.0   # Async testing support` from: thermodynamic.txt

---

### aiohttp
**Conflicting versions:**

- `>=3.12.0` from: base.txt
- `==3.11.16` from: requirements.txt

---

### attrs
**Conflicting versions:**

- `==25.3.0` from: base.txt, requirements.txt
- `>=23.0.0             # Advanced class definitions` from: thermodynamic.txt

---

### pydantic-core
**Conflicting versions:**

- `` from: base.txt
- `==2.20.1` from: requirements.txt

---

### pyyaml
**Conflicting versions:**

- `` from: base.txt
- `==6.0.2` from: requirements.txt

---

### urllib3
**Conflicting versions:**

- `==2.2.2` from: base.txt
- `==2.3.0` from: requirements.txt

---

### zipp
**Conflicting versions:**

- `>=3.20.0` from: base.txt
- `==3.23.0` from: requirements.txt

---

### portalocker
**Conflicting versions:**

- `>=2.0.0` from: base.txt
- `==2.10.1` from: requirements.txt

---

### numpy
**Conflicting versions:**

- `>=2.0.0` from: data.txt, thermodynamic.txt
- `>=1.24.0` from: quantum.txt
- `==2.2.6` from: requirements.txt

---

### scipy
**Conflicting versions:**

- `>=1.10.0` from: data.txt
- `>=1.11.0` from: omnidimensional.txt, quantum.txt, thermodynamic.txt
- `==1.16.0` from: requirements.txt

---

### pandas
**Conflicting versions:**

- `>=2.0.0,<3.0.0` from: data.txt
- `==2.3.0` from: requirements.txt

---

### matplotlib
**Conflicting versions:**

- `>=3.8.0` from: data.txt
- `>=3.7.0` from: quantum.txt
- `==3.10.3` from: requirements.txt
- `>=3.7.0         # Visualization of Fibonacci spirals` from: thermodynamic.txt

---

### networkx
**Conflicting versions:**

- `>=3.0` from: data.txt
- `==3.5` from: requirements.txt

---

### pyparsing
**Conflicting versions:**

- `==3.1.2` from: data.txt
- `==3.2.3` from: requirements.txt

---

### python-dateutil
**Conflicting versions:**

- `==2.9.0` from: data.txt
- `==2.9.0.post0` from: requirements.txt

---

### altair
**Conflicting versions:**

- `>=5.0.0` from: data.txt
- `==5.5.0` from: requirements.txt

---

### dash
**Conflicting versions:**

- `>=3.0.0` from: data.txt
- `==3.0.4` from: requirements.txt
- `>=3.0.0               # Real-time dashboards` from: thermodynamic.txt

---

### dash-bootstrap-components
**Conflicting versions:**

- `>=2.0.0` from: data.txt
- `==2.0.3` from: requirements.txt

---

### flask
**Conflicting versions:**

- `>=3.0.0 # Dependency for Dash` from: data.txt
- `==3.0.3` from: requirements.txt

---

### blinker
**Conflicting versions:**

- `>=1.9.0 # from Flask` from: data.txt
- `==1.9.0` from: requirements.txt

---

### pyarrow
**Conflicting versions:**

- `>=15.0.0` from: data.txt
- `==20.0.0` from: requirements.txt

---

### narwhals
**Conflicting versions:**

- `>=1.0.0 # for polars` from: data.txt
- `==1.42.1` from: requirements.txt

---

### cloudpickle
**Conflicting versions:**

- `>=3.0.0` from: data.txt
- `==3.1.1` from: requirements.txt

---

### dill
**Conflicting versions:**

- `>=0.3.0` from: data.txt
- `==0.3.8` from: requirements.txt

---

### fsspec
**Conflicting versions:**

- `>=2024.1.0` from: data.txt
- `==2025.3.0` from: requirements.txt

---

### joblib
**Conflicting versions:**

- `>=1.3.0` from: data.txt
- `==1.5.1` from: requirements.txt
- `>=1.3.0             # Parallel processing` from: thermodynamic.txt

---

### mmh3
**Conflicting versions:**

- `>=5.0.0` from: data.txt
- `==5.1.0` from: requirements.txt

---

### multiprocess
**Conflicting versions:**

- `>=0.70.0` from: data.txt
- `==0.70.16` from: requirements.txt

---

### pillow
**Conflicting versions:**

- `>=10.0.0` from: data.txt
- `==11.2.1` from: requirements.txt

---

### preshed
**Conflicting versions:**

- `>=3.0.0` from: data.txt
- `==3.0.10` from: requirements.txt

---

### thinc
**Conflicting versions:**

- `>=8.3.0` from: data.txt
- `==8.3.6` from: requirements.txt

---

### wasabi
**Conflicting versions:**

- `>=1.1.0` from: data.txt
- `==1.1.3` from: requirements.txt

---

### hypothesis
**Conflicting versions:**

- `==6.135.12` from: dev.txt, requirements.txt
- `==6.92.1  # Property-based testing` from: omnidimensional.txt
- `>=6.88.0       # Property-based testing` from: thermodynamic.txt

---

### pylint
**Conflicting versions:**

- `` from: dev.txt
- `==3.3.7` from: requirements.txt

---

### black
**Conflicting versions:**

- `` from: dev.txt
- `==25.1.0` from: requirements.txt

---

### line-profiler
**Conflicting versions:**

- `==4.2.0` from: dev.txt, requirements.txt
- `>=4.1.0` from: quantum.txt
- `>=4.1.0      # Code performance profiling` from: thermodynamic.txt

---

### memory-profiler
**Conflicting versions:**

- `==0.61.0` from: dev.txt, omnidimensional.txt, requirements.txt
- `>=0.61.0` from: gpu.txt, quantum.txt
- `>=0.61.0   # Memory usage tracking` from: thermodynamic.txt

---

### gputil
**Conflicting versions:**

- `==1.4.0` from: dev.txt, requirements.txt
- `>=1.4.0` from: gpu.txt

---

### torch
**Conflicting versions:**

- `>=2.1.0` from: gpu.txt, quantum.txt
- `==2.4.0` from: ml.txt
- `>=2.0.0  # Already in gpu.txt but essential for sentiment` from: omnidimensional.txt
- `==2.5.1+cu121` from: requirements.txt
- `>=2.1.0              # Neural networks for consciousness patterns` from: thermodynamic.txt

---

### torchvision
**Conflicting versions:**

- `>=0.16.0` from: gpu.txt
- `` from: ml.txt
- `==0.20.1+cu121` from: requirements.txt

---

### torchaudio
**Conflicting versions:**

- `>=2.1.0` from: gpu.txt
- `` from: ml.txt
- `==2.5.1+cu121` from: requirements.txt

---

### cupy-cuda12x
**Conflicting versions:**

- `>=13.0.0` from: gpu.txt, quantum.txt
- `==13.5.1` from: requirements.txt
- `` from: testing.txt

---

### cuda-python
**Conflicting versions:**

- `>=12.0.0` from: gpu.txt
- `>=12.8.0` from: quantum.txt

---

### pynvml
**Conflicting versions:**

- `>=11.5.0  # NVIDIA Management Library for GPU monitoring` from: gpu.txt
- `>=11.5.0` from: quantum.txt
- `==12.0.0` from: requirements.txt

---

### nvidia-ml-py
**Conflicting versions:**

- `>=7.352.0` from: gpu.txt
- `>=12.0.0` from: quantum.txt
- `==12.575.51` from: requirements.txt

---

### numba
**Conflicting versions:**

- `>=0.58.0` from: gpu.txt
- `==0.61.2` from: requirements.txt
- `>=0.59.0             # JIT compilation for performance` from: thermodynamic.txt

---

### pympler
**Conflicting versions:**

- `>=0.9` from: gpu.txt
- `==1.1` from: requirements.txt

---

### psutil
**Conflicting versions:**

- `>=5.9.0` from: gpu.txt, quantum.txt
- `>=5.9.0  # System monitoring` from: omnidimensional.txt
- `==7.0.0` from: requirements.txt
- `>=5.9.0             # System resource monitoring` from: thermodynamic.txt

---

### transformers
**Conflicting versions:**

- `==4.43.3` from: ml.txt
- `==4.37.2` from: omnidimensional.txt
- `==4.53.0` from: requirements.txt
- `>=4.35.0      # For semantic vector processing` from: thermodynamic.txt

---

### tokenizers
**Conflicting versions:**

- `==0.19.1` from: ml.txt
- `==0.21.2` from: requirements.txt

---

### faiss-cpu
**Conflicting versions:**

- `==1.11.0 # CPU version, faiss-gpu should be in a separate gpu.txt if needed` from: ml.txt
- `==1.11.0` from: requirements.txt

---

### spacy
**Conflicting versions:**

- `==3.7.5` from: ml.txt
- `==3.8.7` from: requirements.txt

---

### srsly
**Conflicting versions:**

- `==2.4.9` from: ml.txt
- `==2.5.1` from: requirements.txt

---

### scikit-learn
**Conflicting versions:**

- `==1.5.1` from: ml.txt
- `==1.4.0` from: omnidimensional.txt
- `>=1.3.0` from: quantum.txt, thermodynamic.txt
- `==1.6.1` from: requirements.txt

---

### safetensors
**Conflicting versions:**

- `==0.4.3` from: ml.txt
- `==0.5.3` from: requirements.txt

---

### sentence-transformers
**Conflicting versions:**

- `==3.0.1` from: ml.txt
- `==3.4.1` from: requirements.txt

---

### web3
**Conflicting versions:**

- `==6.15.1` from: omnidimensional.txt
- `==7.10.0` from: requirements.txt

---

### eth-hash
**Conflicting versions:**

- `==0.6.0` from: omnidimensional.txt
- `==0.7.1` from: requirements.txt

---

### eth-typing
**Conflicting versions:**

- `==4.0.0` from: omnidimensional.txt
- `==5.2.1` from: requirements.txt

---

### eth-utils
**Conflicting versions:**

- `==2.3.1` from: omnidimensional.txt
- `==5.3.0` from: requirements.txt

---

### ccxt
**Conflicting versions:**

- `==4.2.25  # Cryptocurrency trading library` from: omnidimensional.txt
- `==4.4.90` from: requirements.txt, trading.txt

---

### cryptography
**Conflicting versions:**

- `==42.0.2` from: omnidimensional.txt
- `>=42.0.0` from: omnidimensional.txt, trading.txt
- `==45.0.4` from: requirements.txt

---

### ecdsa
**Conflicting versions:**

- `==0.18.0` from: omnidimensional.txt
- `==0.19.1` from: requirements.txt

---

### pycryptodome
**Conflicting versions:**

- `==3.19.0` from: omnidimensional.txt
- `==3.23.0` from: requirements.txt

---

### yfinance
**Conflicting versions:**

- `==0.2.18` from: omnidimensional.txt
- `==0.2.40` from: requirements.txt

---

### alpha-vantage
**Conflicting versions:**

- `==2.3.1` from: omnidimensional.txt
- `==3.0.0` from: requirements.txt, trading.txt

---

### pandas-ta
**Conflicting versions:**

- `==0.3.14b  # Technical analysis indicators` from: omnidimensional.txt
- `==0.3.14b0` from: requirements.txt

---

### textblob
**Conflicting versions:**

- `==0.17.1` from: omnidimensional.txt
- `==0.19.0` from: requirements.txt

---

### websockets
**Conflicting versions:**

- `==12.0` from: omnidimensional.txt, trading.txt
- `==13.1` from: requirements.txt

---

### websocket-client
**Conflicting versions:**

- `==1.7.0` from: omnidimensional.txt
- `==1.8.0` from: requirements.txt

---

### aiolimiter
**Conflicting versions:**

- `==1.1.0` from: omnidimensional.txt
- `==1.2.1` from: requirements.txt

---

### aioredis
**Conflicting versions:**

- `==2.0.1  # For caching and rate limiting` from: omnidimensional.txt
- `==2.0.1` from: requirements.txt
- `>=2.0.0           # Redis async client for caching` from: thermodynamic.txt

---

### cvxpy
**Conflicting versions:**

- `==1.4.1  # Convex optimization` from: omnidimensional.txt
- `>=1.4.0` from: quantum.txt
- `==1.6.6` from: requirements.txt
- `>=1.4.0              # Convex optimization` from: thermodynamic.txt

---

### jsonschema
**Conflicting versions:**

- `==4.20.0` from: omnidimensional.txt
- `==4.24.0` from: requirements.txt

---

### py-spy
**Conflicting versions:**

- `==0.3.14  # Performance profiling` from: omnidimensional.txt
- `==0.4.0` from: requirements.txt

---

### keyring
**Conflicting versions:**

- `==24.3.0` from: omnidimensional.txt
- `==25.6.0` from: requirements.txt

---

### qiskit
**Conflicting versions:**

- `>=1.0.0` from: quantum.txt, thermodynamic.txt
- `==1.2.4` from: requirements.txt
- `` from: testing.txt

---

### qiskit-aer
**Conflicting versions:**

- `>=0.15.0` from: quantum.txt, thermodynamic.txt
- `==0.16.0` from: requirements.txt

---

### cirq
**Conflicting versions:**

- `>=1.4.0` from: quantum.txt
- `==1.5.0` from: requirements.txt

---

### cirq-core
**Conflicting versions:**

- `>=1.4.0` from: quantum.txt
- `==1.5.0` from: requirements.txt

---

### cirq-google
**Conflicting versions:**

- `>=1.4.0` from: quantum.txt
- `==1.5.0` from: requirements.txt

---

### pennylane
**Conflicting versions:**

- `>=0.35.0` from: quantum.txt, thermodynamic.txt
- `==0.41.1` from: requirements.txt

---

### pennylane-lightning
**Conflicting versions:**

- `>=0.35.0` from: quantum.txt
- `==0.41.1` from: requirements.txt

---

### sympy
**Conflicting versions:**

- `>=1.12.0` from: quantum.txt, thermodynamic.txt
- `==1.14.0` from: requirements.txt

---

### opt-einsum
**Conflicting versions:**

- `>=3.3.0` from: quantum.txt
- `==3.4.0` from: requirements.txt

---

### nlopt
**Conflicting versions:**

- `>=2.7.0` from: quantum.txt
- `>=2.7.0              # Non-linear optimization library` from: thermodynamic.txt

---

### seaborn
**Conflicting versions:**

- `>=0.13.0` from: quantum.txt
- `==0.13.2` from: requirements.txt
- `>=0.13.0           # Statistical visualization` from: thermodynamic.txt

---

### plotly
**Conflicting versions:**

- `>=5.17.0` from: quantum.txt
- `==6.1.2` from: requirements.txt
- `>=5.17.0            # Interactive thermodynamic visualizations` from: thermodynamic.txt

---

### colorlog
**Conflicting versions:**

- `==6.9.0` from: requirements.txt
- `>=6.8.0           # Colored log output` from: thermodynamic.txt

---

### confluent-kafka
**Conflicting versions:**

- `==2.10.1` from: requirements.txt
- `` from: testing.txt

---

### cython
**Conflicting versions:**

- `==3.1.2` from: requirements.txt
- `>=3.0.0             # C extensions for speed` from: thermodynamic.txt

---

### mpmath
**Conflicting versions:**

- `==1.3.0` from: requirements.txt
- `>=1.3.0             # High-precision arithmetic for golden ratio` from: thermodynamic.txt

---

### pytest-mock
**Conflicting versions:**

- `==3.14.1` from: requirements.txt
- `` from: testing.txt

---

### pywavelets
**Conflicting versions:**

- `==1.8.0` from: requirements.txt
- `>=1.4.0         # Wavelet transforms for signal analysis` from: thermodynamic.txt

---

### rich
**Conflicting versions:**

- `==14.0.0` from: requirements.txt
- `>=13.7.0              # Rich text and progress bars` from: thermodynamic.txt

---

### structlog
**Conflicting versions:**

- `==25.4.0` from: requirements.txt
- `>=23.2.0         # Structured logging` from: thermodynamic.txt

---
