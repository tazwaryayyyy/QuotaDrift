"""
Advanced Model Management with Circuit Breaker and Dynamic Scoring.

Features:
- Circuit breaker per model with configurable failure thresholds
- Dynamic model scoring based on latency, success rate, and load
- Intelligent model selection with weighted scoring
- Graceful degradation and fallback handling
- Structured logging with request tracing
"""

import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import asyncio
from functools import lru_cache

import config

# Import metrics (will be imported after main.py initializes them)
try:
    from main import MODEL_REQUESTS, MODEL_LATENCY, TOKEN_USAGE
except ImportError:
    # Fallback dummy metrics for testing
    class DummyMetric:
        def labels(self, **kwargs): return self
        def inc(self): pass
        def observe(self, value): pass
    MODEL_REQUESTS = DummyMetric()
    MODEL_LATENCY = DummyMetric()
    TOKEN_USAGE = DummyMetric()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_manager")

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration per model."""
    failure_threshold: int = 3      # Failures before opening
    recovery_timeout: int = 300    # Seconds before trying recovery
    half_open_max_calls: int = 2    # Max calls in half-open state


@dataclass
class ModelMetrics:
    """Detailed metrics for each model."""
    model_id: str
    slot_name: str
    
    # Circuit breaker state
    circuit_state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    
    # Performance metrics
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Usage tracking
    total_requests: int = 0
    total_errors: int = 0
    last_used: Optional[datetime] = None
    
    # Rate limiting
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    
    # Scoring factors
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    load_score: float = 0.0
    priority_score: float = 1.0


class CircuitBreaker:
    """Circuit breaker implementation for individual models."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
    
    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        now = datetime.utcnow()
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.next_attempt_time and now >= self.next_attempt_time:
                self.state = "half_open"
                logger.info(f"Circuit breaker transitioning to half-open")
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record a successful execution."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful execution")
    
    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == "closed":
            if self.failure_count >= self.config.failure_threshold:
                self.state = "open"
                self.next_attempt_time = datetime.utcnow() + timedelta(
                    seconds=self.config.recovery_timeout
                )
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == "half_open":
            self.state = "open"
            self.next_attempt_time = datetime.utcnow() + timedelta(
                seconds=self.config.recovery_timeout
            )
            logger.warning("Circuit breaker re-opened from half-open state")


class ModelManager:
    """Advanced model management with circuit breaker and dynamic scoring."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.request_traces: Dict[str, dict] = {}
        
        # Initialize circuit breakers and metrics for all models
        self._initialize_models()
        
        # Note: Background tasks will be started when the event loop is running
        self._background_tasks_started = False
    
    def _initialize_models(self):
        """Initialize circuit breakers and metrics for all configured models."""
        cb_config = CircuitBreakerConfig()
        
        for slot in config.MODEL_LIST:
            slot_name = slot["model_name"]
            model_id = slot["litellm_params"]["model"]
            
            self.circuit_breakers[slot_name] = CircuitBreaker(cb_config)
            self.metrics[slot_name] = ModelMetrics(
                model_id=model_id,
                slot_name=slot_name,
                priority_score=self._get_priority_score(slot_name)
            )
    
    def _get_priority_score(self, slot_name: str) -> float:
        """Get priority score based on model position in chain."""
        priority_map = {
            "primary": 1.0,
            "primary_b": 0.9,
            "secondary": 0.8,
            "tertiary": 0.7,
            "quaternary": 0.6,
            "fallback": 0.5
        }
        return priority_map.get(slot_name, 0.5)
    
    def get_available_models(self) -> List[str]:
        """Get list of models that can currently accept requests."""
        available = []
        
        for slot_name, metrics in self.metrics.items():
            cb = self.circuit_breakers[slot_name]
            
            # Check circuit breaker
            if not cb.can_execute():
                continue
            
            # Check rate limits
            if metrics.rate_limit_remaining == 0:
                if metrics.rate_limit_reset and datetime.utcnow() < metrics.rate_limit_reset:
                    continue
            
            available.append(slot_name)
        
        return available
    
    def get_best_model(self, request_id: str) -> Optional[str]:
        """Select the best model based on dynamic scoring."""
        available = self.get_available_models()
        
        if not available:
            logger.warning(f"No models available for request {request_id}")
            return None
        
        # Calculate scores for available models
        scored_models = []
        for slot_name in available:
            score = self._calculate_model_score(slot_name)
            scored_models.append((score, slot_name))
        
        # Sort by score (highest first)
        scored_models.sort(reverse=True)
        
        best_model = scored_models[0][1]
        logger.info(f"Selected model {best_model} with score {scored_models[0][2]:.3f} for request {request_id}")
        
        return best_model
    
    def _calculate_model_score(self, slot_name: str) -> Tuple[float, float, float, float]:
        """Calculate comprehensive score for a model."""
        metrics = self.metrics[slot_name]
        
        # Success rate score (0-1)
        success_score = metrics.success_rate
        
        # Latency score (inverse, 0-1)
        latency_score = 1.0 / (1.0 + metrics.avg_latency_ms / 1000.0)
        
        # Load score (based on recent usage)
        load_score = max(0, 1.0 - metrics.load_score)
        
        # Priority score (pre-defined)
        priority_score = metrics.priority_score
        
        # Weighted combination
        total_score = (
            success_score * 0.4 +
            latency_score * 0.3 +
            load_score * 0.2 +
            priority_score * 0.1
        )
        
        return total_score, success_score, latency_score, load_score
    
    def start_request(self, slot_name: str, request_id: str) -> dict:
        """Start tracking a request."""
        trace = {
            "request_id": request_id,
            "slot_name": slot_name,
            "start_time": time.monotonic(),
            "start_datetime": datetime.utcnow(),
        }
        self.request_traces[request_id] = trace
        
        metrics = self.metrics[slot_name]
        metrics.total_requests += 1
        metrics.last_used = datetime.utcnow()
        
        logger.info(f"Started request {request_id} on model {slot_name}")
        
        return trace
    
    def record_success(self, slot_name: str, request_id: str, tokens: int = 0):
        """Record a successful request completion."""
        if request_id not in self.request_traces:
            return
        
        trace = self.request_traces[request_id]
        end_time = time.monotonic()
        latency_ms = (end_time - trace["start_time"]) * 1000
        
        # Update circuit breaker
        self.circuit_breakers[slot_name].record_success()
        
        # Update metrics
        metrics = self.metrics[slot_name]
        metrics.recent_successes.append(end_time)
        metrics.recent_latencies.append(latency_ms)
        
        # Update Prometheus metrics
        MODEL_REQUESTS.labels(model=metrics.model_id, status='success').inc()
        MODEL_LATENCY.labels(model=metrics.model_id).observe(latency_ms / 1000.0)
        if tokens > 0:
            TOKEN_USAGE.labels(model=metrics.model_id).inc(tokens)
        
        # Update rolling averages
        self._update_metrics(slot_name)
        
        # Clean up trace
        del self.request_traces[request_id]
        
        logger.info(f"Request {request_id} completed successfully in {latency_ms:.1f}ms")
    
    def record_failure(self, slot_name: str, request_id: str, error: str):
        """Record a failed request."""
        if request_id not in self.request_traces:
            return
        
        trace = self.request_traces[request_id]
        end_time = time.monotonic()
        
        # Update circuit breaker
        self.circuit_breakers[slot_name].record_failure()
        
        # Update metrics
        metrics = self.metrics[slot_name]
        metrics.total_errors += 1
        metrics.recent_failures.append(end_time)
        
        # Update Prometheus metrics
        MODEL_REQUESTS.labels(model=metrics.model_id, status='error').inc()
        
        # Update rolling averages
        self._update_metrics(slot_name)
        
        # Clean up trace
        del self.request_traces[request_id]
        
        logger.error(f"Request {request_id} failed on model {slot_name}: {error}")
    
    def update_rate_limit(self, slot_name: str, remaining: Optional[int], reset: Optional[str]):
        """Update rate limit information."""
        metrics = self.metrics[slot_name]
        metrics.rate_limit_remaining = remaining
        
        if reset:
            try:
                metrics.rate_limit_reset = datetime.fromisoformat(reset.replace('Z', '+00:00'))
            except:
                pass
    
    def _update_metrics(self, slot_name: str):
        """Update rolling metrics for a model."""
        metrics = self.metrics[slot_name]
        
        # Calculate success rate
        total_recent = len(metrics.recent_successes) + len(metrics.recent_failures)
        if total_recent > 0:
            metrics.success_rate = len(metrics.recent_successes) / total_recent
        
        # Calculate average latency
        if metrics.recent_latencies:
            metrics.avg_latency_ms = sum(metrics.recent_latencies) / len(metrics.recent_latencies)
        
        # Calculate load score (requests per minute)
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = sum(
            1 for success_time in metrics.recent_successes 
            if success_time >= one_minute_ago.timestamp()
        )
        metrics.load_score = min(1.0, recent_requests / 10.0)  # Normalize to 0-1
    
    def start_background_tasks(self):
        """Start background tasks when event loop is available."""
        if not self._background_tasks_started:
            asyncio.create_task(self._cleanup_old_traces())
            asyncio.create_task(self._update_scores())
            self._background_tasks_started = True
    
    async def _update_scores(self):
        """Background task to update model scores periodically."""
        while True:
            try:
                for slot_name in self.metrics:
                    self._update_metrics(slot_name)
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error updating scores: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_traces(self):
        """Background task to clean up old request traces."""
        while True:
            try:
                now = time.monotonic()
                expired = [
                    req_id for req_id, trace in self.request_traces.items()
                    if now - trace["start_time"] > 300  # 5 minutes
                ]
                
                for req_id in expired:
                    del self.request_traces[req_id]
                
                await asyncio.sleep(60)  # Clean up every minute
            except Exception as e:
                logger.error(f"Error cleaning up traces: {e}")
                await asyncio.sleep(60)
    
    def get_health_snapshot(self) -> List[dict]:
        """Get comprehensive health snapshot for all models."""
        snapshot = []
        
        for slot_name, metrics in self.metrics.items():
            cb = self.circuit_breakers[slot_name]
            
            # Determine overall health status
            if cb.state == "open":
                status = "failed"
            elif cb.state == "half_open":
                status = "cooling"
            elif metrics.success_rate < 0.5:
                status = "degraded"
            else:
                status = "available"
            
            snapshot.append({
                "id": slot_name,
                "model_id": metrics.model_id,
                "display": config.MODEL_DISPLAY.get(metrics.model_id, metrics.model_id),
                "color": config.MODEL_COLORS.get(metrics.model_id, "#94a3b8"),
                "status": status,
                "circuit_state": cb.state,
                "requests": metrics.total_requests,
                "errors": metrics.total_errors,
                "success_rate": metrics.success_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
                "rate_limit_remaining": metrics.rate_limit_remaining,
                "rate_limit_reset": metrics.rate_limit_reset.isoformat() if metrics.rate_limit_reset else None,
                "score": self._calculate_model_score(slot_name)[0],
            })
        
        return snapshot


# Global model manager instance
model_manager = ModelManager()

def get_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

@lru_cache(maxsize=128)
def get_model_color(model_id: str) -> str:
    """Get model color with caching."""
    return config.MODEL_COLORS.get(model_id, "#94a3b8")
