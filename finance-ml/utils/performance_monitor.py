"""
Day 5: Performance Monitoring and Optimization
utils/performance_monitor.py

Monitors pipeline performance and provides optimization recommendations
"""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_percent: float
    active_threads: int
    processing_rate: float = 0.0
    queue_size: int = 0

@dataclass
class StagePerformance:
    """Performance metrics for a pipeline stage"""
    stage_name: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    
    @property
    def average_time(self) -> float:
        return self.total_time / max(self.execution_count, 1)
    
    def update(self, execution_time: float, success: bool):
        """Update stage performance metrics"""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        
        if not success:
            self.error_count += 1
        
        self.success_rate = ((self.execution_count - self.error_count) / self.execution_count) * 100

class PerformanceMonitor:
    """Real-time performance monitoring for the pipeline"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.system_metrics: List[PerformanceMetrics] = []
        self.stage_performance: Dict[str, StagePerformance] = {}
        self.processing_events: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'processing_rate_min': 0.5,  # companies per second
            'average_time_max': 30.0     # seconds per company
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.system_metrics.append(metrics)
                    
                    # Keep only last 1000 measurements
                    if len(self.system_metrics) > 1000:
                        self.system_metrics = self.system_metrics[-1000:]
                
                # Check thresholds and alert if needed
                self._check_thresholds(metrics)
                
            except Exception as e:
                logger.warning(f"Error in performance monitoring: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / (1024 * 1024)  # MB
            memory_percent = memory_info.percent
            
            # Thread count
            active_threads = threading.active_count()
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_percent=memory_percent,
                active_threads=active_threads
            )
            
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=0, memory_usage=0, memory_percent=0, active_threads=1
            )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed warning thresholds"""
        
        # CPU warnings
        if metrics.cpu_usage > self.thresholds['cpu_critical']:
            logger.critical(f"CPU usage critical: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > self.thresholds['cpu_warning']:
            logger.warning(f"CPU usage high: {metrics.cpu_usage:.1f}%")
        
        # Memory warnings
        if metrics.memory_percent > self.thresholds['memory_critical']:
            logger.critical(f"Memory usage critical: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            logger.warning(f"Memory usage high: {metrics.memory_percent:.1f}%")
    
    def log_processing_event(self, event_type: str, company_id: str, 
                           stage: str, duration: float, success: bool):
        """Log a processing event"""
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'company_id': company_id,
            'stage': stage,
            'duration': duration,
            'success': success
        }
        
        with self.lock:
            self.processing_events.append(event)
            
            # Keep only last 10000 events
            if len(self.processing_events) > 10000:
                self.processing_events = self.processing_events[-10000:]
            
            # Update stage performance
            if stage not in self.stage_performance:
                self.stage_performance[stage] = StagePerformance(stage_name=stage)
            
            self.stage_performance[stage].update(duration, success)
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot"""
        
        with self.lock:
            if not self.system_metrics:
                return {'status': 'no_data'}
            
            # Latest system metrics
            latest = self.system_metrics[-1]
            
            # Recent processing rate
            recent_events = [e for e in self.processing_events 
                           if time.time() - e['timestamp'] < 60]  # Last minute
            processing_rate = len(recent_events) / 60.0
            
            # Stage performance summary
            stage_summary = {}
            for stage, perf in self.stage_performance.items():
                stage_summary[stage] = {
                    'average_time': perf.average_time,
                    'success_rate': perf.success_rate,
                    'execution_count': perf.execution_count,
                    'error_count': perf.error_count
                }
            
            return {
                'timestamp': latest.timestamp,
                'system': {
                    'cpu_usage': latest.cpu_usage,
                    'memory_usage': latest.memory_usage,
                    'memory_percent': latest.memory_percent,
                    'active_threads': latest.active_threads
                },
                'processing': {
                    'rate_per_minute': processing_rate * 60,
                    'recent_events': len(recent_events)
                },
                'stages': stage_summary
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        with self.lock:
            if not self.system_metrics:
                return {'status': 'no_data'}
            
            # Calculate system statistics
            cpu_values = [m.cpu_usage for m in self.system_metrics[-100:]]  # Last 100 samples
            memory_values = [m.memory_percent for m in self.system_metrics[-100:]]
            
            system_stats = {
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': statistics.mean(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0,
                    'min': min(cpu_values) if cpu_values else 0
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': statistics.mean(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0,
                    'min': min(memory_values) if memory_values else 0
                }
            }
            
            # Processing statistics
            successful_events = [e for e in self.processing_events if e['success']]
            failed_events = [e for e in self.processing_events if not e['success']]
            
            processing_stats = {
                'total_events': len(self.processing_events),
                'successful_events': len(successful_events),
                'failed_events': len(failed_events),
                'success_rate': (len(successful_events) / max(len(self.processing_events), 1)) * 100,
                'average_processing_time': statistics.mean([e['duration'] for e in successful_events]) if successful_events else 0
            }
            
            # Stage performance
            stage_stats = {}
            for stage, perf in self.stage_performance.items():
                stage_stats[stage] = {
                    'execution_count': perf.execution_count,
                    'average_time': perf.average_time,
                    'min_time': perf.min_time if perf.min_time != float('inf') else 0,
                    'max_time': perf.max_time,
                    'success_rate': perf.success_rate,
                    'error_count': perf.error_count
                }
            
            return {
                'monitoring_duration': time.time() - self.system_metrics[0].timestamp if self.system_metrics else 0,
                'system_performance': system_stats,
                'processing_performance': processing_stats,
                'stage_performance': stage_stats,
                'recommendations': self._generate_recommendations(system_stats, processing_stats, stage_stats)
            }
    
    def _generate_recommendations(self, system_stats: Dict, processing_stats: Dict, 
                                stage_stats: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if system_stats['cpu']['average'] > 80:
            recommendations.append("Consider reducing parallel workers or batch size to lower CPU usage")
        elif system_stats['cpu']['average'] < 30:
            recommendations.append("CPU utilization is low - consider increasing parallel workers")
        
        # Memory recommendations
        if system_stats['memory']['average'] > 80:
            recommendations.append("High memory usage detected - consider processing smaller batches")
        
        # Processing speed recommendations
        if processing_stats['average_processing_time'] > 10:
            recommendations.append("Processing time is high - check for bottlenecks in API or database operations")
        
        # Success rate recommendations
        if processing_stats['success_rate'] < 90:
            recommendations.append("Low success rate detected - review error handling and retry logic")
        
        # Stage-specific recommendations
        for stage, stats in stage_stats.items():
            if stats['success_rate'] < 85:
                recommendations.append(f"Stage '{stage}' has low success rate ({stats['success_rate']:.1f}%) - needs investigation")
            
            if stats['average_time'] > 15:
                recommendations.append(f"Stage '{stage}' is slow (avg: {stats['average_time']:.1f}s) - consider optimization")
        
        if not recommendations:
            recommendations.append("Performance looks good - no specific recommendations")
        
        return recommendations
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        try:
            import json
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_performance_summary(),
                'current': self.get_current_performance()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Performance metrics exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""

class PerformanceProfiler:
    """Context manager for profiling code sections"""
    
    def __init__(self, monitor: PerformanceMonitor, stage_name: str, 
                 company_id: str = None):
        self.monitor = monitor
        self.stage_name = stage_name
        self.company_id = company_id
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.success = exc_type is None
        
        self.monitor.log_processing_event(
            event_type='stage_execution',
            company_id=self.company_id or 'unknown',
            stage=self.stage_name,
            duration=duration,
            success=self.success
        )

# Factory functions
def create_performance_monitor(monitoring_interval: float = 1.0) -> PerformanceMonitor:
    """Create and return a performance monitor instance"""
    return PerformanceMonitor(monitoring_interval=monitoring_interval)

def profile_stage(monitor: PerformanceMonitor, stage_name: str, company_id: str = None):
    """Create a performance profiler for a stage"""
    return PerformanceProfiler(monitor, stage_name, company_id)

# Example usage
if __name__ == "__main__":
    # Example of using the performance monitor
    monitor = create_performance_monitor()
    monitor.start_monitoring()
    
    # Simulate some processing
    for i in range(5):
        with profile_stage(monitor, 'test_stage', f'COMPANY_{i}'):
            time.sleep(1)  # Simulate work
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print("Performance Summary:")
    print(f"Total events: {summary['processing_performance']['total_events']}")
    print(f"Success rate: {summary['processing_performance']['success_rate']:.1f}%")
    
    monitor.stop_monitoring()