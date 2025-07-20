"""
Kimera Prometheus Metrics
========================

Comprehensive Prometheus metrics collection for all Kimera SWM components.
Provides the metrics that Grafana dashboards expect to find.
"""

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
import time
import psutil
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_metrics_instance = None
_metrics_lock = threading.Lock()

class KimeraPrometheusMetrics:
    """
    Centralized Prometheus metrics for Kimera SWM system.
    
    Provides all the metrics that Grafana dashboards expect to see.
    """
    
    # Thread-safe singleton implementation to avoid duplicated timeseries
    _singleton_lock: threading.Lock = threading.Lock()
    _singleton_instance: "KimeraPrometheusMetrics | None" = None

    def __new__(cls, *args, **kwargs):  # noqa: D401
        if cls._singleton_instance is None:
            with cls._singleton_lock:
                if cls._singleton_instance is None:
                    cls._singleton_instance = super().__new__(cls)
        return cls._singleton_instance

    def __init__(self):
        # Skip re-initialisation if we already set up metrics
        if getattr(self, "_initialized", False):
            return

        # Core Kimera Metrics
        self.kimera_geoids_total = Counter(
            'kimera_geoids_total', 
            'Total number of geoids created',
            ['vault', 'type']
        )
        
        self.kimera_scars_total = Counter(
            'kimera_scars_total', 
            'Total number of SCARs formed',
            ['type', 'vault']
        )
        
        self.kimera_revolutionary_insights_total = Counter(
            'kimera_revolutionary_insights_total', 
            'Total revolutionary insights generated'
        )
        
        self.kimera_cognitive_cycles_total = Gauge(
            'kimera_cognitive_cycles_total', 
            'Total cognitive cycles completed'
        )
        
        self.kimera_contradictions_total = Counter(
            'kimera_contradictions_total', 
            'Total contradictions detected',
            ['resolution_type']
        )
        
        # API and System Metrics
        self.kimera_requests_total = Counter(
            'kimera_requests_total', 
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.kimera_errors_total = Counter(
            'kimera_errors_total', 
            'Total errors by component',
            ['component', 'error_type']
        )
        
        self.kimera_selective_feedback_ops_total = Counter(
            'kimera_selective_feedback_ops_total', 
            'Total selective feedback operations',
            ['operation_type', 'accuracy_level']
        )
        
        # Performance Metrics
        self.kimera_request_duration_seconds = Histogram(
            'kimera_request_duration_seconds', 
            'Request latency in seconds',
            ['endpoint', 'method'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # GPU Metrics
        self.kimera_gpu_utilization_percent = Gauge(
            'kimera_gpu_utilization_percent', 
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        self.kimera_gpu_temperature_celsius = Gauge(
            'kimera_gpu_temperature_celsius', 
            'GPU temperature in Celsius',
            ['gpu_id']
        )
        
        self.kimera_gpu_memory_used_bytes = Gauge(
            'kimera_gpu_memory_used_bytes', 
            'GPU memory used in bytes',
            ['gpu_id']
        )
        
        # System Resource Metrics
        self.kimera_system_memory_bytes = Gauge(
            'kimera_system_memory_bytes', 
            'System memory usage in bytes',
            ['type']  # used, available, total
        )
        
        self.kimera_system_cpu_percent = Gauge(
            'kimera_system_cpu_percent', 
            'System CPU usage percentage'
        )
        
        self.kimera_system_network_bytes_total = Counter(
            'kimera_system_network_bytes_total', 
            'Network bytes transferred',
            ['direction']  # sent, received
        )
        
        # Cognitive Architecture Metrics
        self.kimera_cognitive_coherence = Gauge(
            'kimera_cognitive_coherence', 
            'Cognitive coherence score (0-1)'
        )
        
        self.kimera_entropy_shannon = Gauge(
            'kimera_entropy_shannon', 
            'Shannon entropy of the system'
        )
        
        self.kimera_entropy_thermodynamic = Gauge(
            'kimera_entropy_thermodynamic', 
            'Thermodynamic entropy of the system'
        )
        
        self.kimera_understanding_quality = Gauge(
            'kimera_understanding_quality', 
            'Understanding quality score',
            ['component']
        )
        
        # Engine Status Metrics
        self.kimera_engine_status = Gauge(
            'kimera_engine_status', 
            'Engine operational status (1=operational, 0=down)',
            ['engine_name']
        )
        
        # Selective Feedback Metrics
        self.kimera_selective_feedback_accuracy = Gauge(
            'kimera_selective_feedback_accuracy', 
            'Selective feedback accuracy score',
            ['feedback_type']
        )
        
        # Embedding Generation Metrics
        self.kimera_embedding_generation_duration_seconds = Histogram(
            'kimera_embedding_generation_duration_seconds',
            'Embedding generation latency in seconds',
            ['model_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Cognitive Field Metrics
        self.kimera_field_resonance_amplitude = Gauge(
            'kimera_field_resonance_amplitude',
            'Cognitive field resonance amplitude',
            ['field_type']
        )
        
        self.kimera_field_coherence_score = Gauge(
            'kimera_field_coherence_score',
            'Field coherence measurement',
            ['field_dimension']
        )
        
        # Service Info
        self.kimera_service_info = Info(
            'kimera_service_info', 
            'Kimera service information'
        )
        
        # Initialize service info
        self.kimera_service_info.info({
            'version': '0.1.0',
            'mode': 'understanding',
            'gpu_enabled': 'true',
            'started_at': datetime.now().isoformat()
        })
        
        logger.info("🎯 Kimera Prometheus metrics initialized")
        self._initialized = True
    
    def record_geoid_created(self, vault: str = "default", geoid_type: str = "semantic"):
        """Record a geoid creation"""
        self.kimera_geoids_total.labels(vault=vault, type=geoid_type).inc()
    
    def record_scar_formed(self, scar_type: str = "contradiction", vault: str = "default"):
        """Record a SCAR formation"""
        self.kimera_scars_total.labels(type=scar_type, vault=vault).inc()
    
    def record_revolutionary_insight(self):
        """Record a revolutionary insight generation"""
        self.kimera_revolutionary_insights_total.inc()
    
    def record_contradiction(self, resolution_type: str = "collapse"):
        """Record a contradiction detection and resolution"""
        try:
            self.kimera_contradictions_total.labels(resolution_type=resolution_type).inc()
            # Get current value for debugging
            current_value = self.kimera_contradictions_total.labels(resolution_type=resolution_type)._value._value
            logger.info(f"✅ RECORDED CONTRADICTION: {resolution_type} (Total: {current_value})")
            logger.info(f"✅ METRICS: Recorded contradiction {resolution_type}, total: {current_value}")
        except Exception as e:
            logger.error(f"❌ ERROR RECORDING CONTRADICTION: {e}")
            logger.error(f"❌ METRICS ERROR: {e}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an API request"""
        status = f"{status_code}"
        self.kimera_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.kimera_request_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)
    
    def record_error(self, component: str, error_type: str):
        """Record an error"""
        self.kimera_errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_selective_feedback(self, operation_type: str, accuracy_level: str):
        """Record selective feedback operation"""
        self.kimera_selective_feedback_ops_total.labels(
            operation_type=operation_type, 
            accuracy_level=accuracy_level
        ).inc()
    
    def update_cognitive_cycles(self, count: int):
        """Update cognitive cycles count"""
        self.kimera_cognitive_cycles_total.set(count)
    
    def update_cognitive_coherence(self, coherence: float):
        """Update cognitive coherence score"""
        self.kimera_cognitive_coherence.set(coherence)
    
    def update_entropy_metrics(self, shannon_entropy: float, thermodynamic_entropy: float):
        """Update entropy metrics"""
        self.kimera_entropy_shannon.set(shannon_entropy)
        self.kimera_entropy_thermodynamic.set(thermodynamic_entropy)
    
    def update_understanding_quality(self, component: str, quality: float):
        """Update understanding quality for a component"""
        self.kimera_understanding_quality.labels(component=component).set(quality)
    
    def update_engine_status(self, engine_name: str, is_operational: bool):
        """Update engine operational status"""
        self.kimera_engine_status.labels(engine_name=engine_name).set(1 if is_operational else 0)
    
    def update_selective_feedback_accuracy(self, feedback_type: str, accuracy: float):
        """Update selective feedback accuracy"""
        self.kimera_selective_feedback_accuracy.labels(feedback_type=feedback_type).set(accuracy)
    
    def record_embedding_generation(self, model_type: str, duration: float):
        """Record embedding generation latency"""
        self.kimera_embedding_generation_duration_seconds.labels(model_type=model_type).observe(duration)
    
    def update_field_resonance_amplitude(self, field_type: str, amplitude: float):
        """Update cognitive field resonance amplitude"""
        self.kimera_field_resonance_amplitude.labels(field_type=field_type).set(amplitude)
    
    def update_field_coherence_score(self, field_dimension: str, coherence: float):
        """Update field coherence measurement"""
        self.kimera_field_coherence_score.labels(field_dimension=field_dimension).set(coherence)
    
    def update_gpu_metrics(self):
        """Update GPU metrics (if available)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.kimera_gpu_utilization_percent.labels(gpu_id=str(i)).set(utilization.gpu)
                
                # GPU temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self.kimera_gpu_temperature_celsius.labels(gpu_id=str(i)).set(temperature)
                
                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.kimera_gpu_memory_used_bytes.labels(gpu_id=str(i)).set(memory_info.used)
                
        except ImportError:
            logger.debug("pynvml not available, skipping GPU metrics")
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.kimera_system_memory_bytes.labels(type="used").set(memory.used)
            self.kimera_system_memory_bytes.labels(type="available").set(memory.available)
            self.kimera_system_memory_bytes.labels(type="total").set(memory.total)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.kimera_system_cpu_percent.set(cpu_percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.kimera_system_network_bytes_total.labels(direction="sent").inc(network.bytes_sent)
            self.kimera_system_network_bytes_total.labels(direction="received").inc(network.bytes_recv)
            
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
    
    def start_background_collection(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                try:
                    self.update_gpu_metrics()
                    self.update_system_metrics()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Error in background metrics collection: {e}")
                    time.sleep(10)  # Wait longer on error
        
        self.collection_thread = threading.Thread(target=collect_metrics, daemon=True)
        self.collection_thread.start()
        logger.info("🔄 Background metrics collection started")

    def stop_background_collection(self):
        """Stops the background metric collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_event.set()
            self.collection_thread.join()
            logger.info("🛑 Stopped background metrics collection")

def get_kimera_metrics() -> "KimeraPrometheusMetrics":
    """
    Returns a singleton instance of the KimeraPrometheusMetrics class.
    
    This function ensures that metrics are registered with Prometheus only once
    by using a thread-safe singleton pattern.
    """
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                logger.info("Instantiating singleton KimeraPrometheusMetrics...")
                _metrics_instance = KimeraPrometheusMetrics()
    return _metrics_instance

def initialize_background_collection():
    """Initializes and starts the background metric collection thread."""
    metrics = get_kimera_metrics()
    metrics.start_background_collection()

# Generate Prometheus metrics string
def generate_metrics_report() -> str:
    """Generates the Prometheus metrics report."""
    return generate_latest()

# Content-type for Prometheus endpoint
PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST 