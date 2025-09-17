"""
Day 5: Batch Processing Optimizer
utils/batch_optimizer.py

Optimizes batch processing for 100 companies with dynamic adjustment
"""

import time
import logging
import statistics
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 10
    max_workers: int = 3
    timeout_per_company: float = 300.0  # 5 minutes
    retry_failed_companies: bool = True
    adaptive_sizing: bool = True
    min_batch_size: int = 2
    max_batch_size: int = 20

@dataclass
class BatchMetrics:
    """Metrics for batch performance"""
    batch_number: int
    companies_count: int
    successful_count: int
    failed_count: int
    processing_time: float
    average_time_per_company: float
    throughput: float  # companies per second
    
    @property
    def success_rate(self) -> float:
        return (self.successful_count / max(self.companies_count, 1)) * 100

class AdaptiveBatchOptimizer:
    """Adaptive batch size optimizer based on performance"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.batch_history: List[BatchMetrics] = []
        self.lock = threading.Lock()
        
        # Performance targets
        self.target_throughput = 2.0  # companies per second
        self.target_success_rate = 95.0  # percent
        self.max_processing_time = 60.0  # seconds per batch
        
    def optimize_batch_size(self, current_performance: List[BatchMetrics]) -> int:
        """Optimize batch size based on recent performance"""
        if not current_performance or not self.config.adaptive_sizing:
            return self.config.batch_size
        
        recent_batches = current_performance[-3:]  # Last 3 batches
        
        if len(recent_batches) < 2:
            return self.config.batch_size
        
        # Calculate performance metrics
        avg_throughput = statistics.mean([b.throughput for b in recent_batches])
        avg_success_rate = statistics.mean([b.success_rate for b in recent_batches])
        avg_processing_time = statistics.mean([b.processing_time for b in recent_batches])
        
        current_batch_size = recent_batches[-1].companies_count
        
        # Decision logic
        if avg_success_rate < self.target_success_rate:
            # Low success rate - reduce batch size
            new_size = max(current_batch_size - 2, self.config.min_batch_size)
            logger.info(f"Reducing batch size to {new_size} due to low success rate ({avg_success_rate:.1f}%)")
            
        elif avg_processing_time > self.max_processing_time:
            # Processing too slow - reduce batch size
            new_size = max(current_batch_size - 1, self.config.min_batch_size)
            logger.info(f"Reducing batch size to {new_size} due to slow processing ({avg_processing_time:.1f}s)")
            
        elif avg_throughput < self.target_throughput and current_batch_size < self.config.max_batch_size:
            # Low throughput - try increasing batch size
            new_size = min(current_batch_size + 1, self.config.max_batch_size)
            logger.info(f"Increasing batch size to {new_size} to improve throughput ({avg_throughput:.2f} c/s)")
            
        else:
            # Performance is acceptable - maintain current size
            new_size = current_batch_size
        
        return new_size
    
    def optimize_worker_count(self, current_performance: List[BatchMetrics]) -> int:
        """Optimize number of workers based on performance"""
        if not current_performance:
            return self.config.max_workers
        
        recent_batch = current_performance[-1]
        
        # If success rate is low, reduce workers to avoid overwhelming systems
        if recent_batch.success_rate < 80:
            return max(1, self.config.max_workers - 1)
        
        # If processing is very fast, we might be able to handle more workers
        if recent_batch.average_time_per_company < 5.0:
            return min(self.config.max_workers + 1, 5)
        
        return self.config.max_workers

class EnhancedBatchProcessor:
    """Enhanced batch processor with optimization and monitoring"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.optimizer = AdaptiveBatchOptimizer(config)
        self.batch_metrics: List[BatchMetrics] = []
        self.failed_companies: List[str] = []
        self.lock = threading.Lock()
        
    def process_companies_optimized(self, 
                                  companies: List[str], 
                                  process_function: Callable[[str], Dict[str, Any]],
                                  progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Process companies with optimized batching
        """
        logger.info(f"Starting optimized batch processing for {len(companies)} companies")
        
        all_results = []
        remaining_companies = companies.copy()
        batch_number = 0
        
        while remaining_companies:
            batch_number += 1
            
            # Optimize batch size based on performance
            if batch_number > 1:
                optimal_batch_size = self.optimizer.optimize_batch_size(self.batch_metrics)
                optimal_workers = self.optimizer.optimize_worker_count(self.batch_metrics)
            else:
                optimal_batch_size = self.config.batch_size
                optimal_workers = self.config.max_workers
            
            # Create current batch
            current_batch = remaining_companies[:optimal_batch_size]
            remaining_companies = remaining_companies[optimal_batch_size:]
            
            logger.info(f"\n--- BATCH {batch_number} ---")
            logger.info(f"Companies: {len(current_batch)}, Workers: {optimal_workers}")
            logger.info(f"Processing: {current_batch}")
            
            # Process batch
            batch_results, batch_metrics = self._process_single_batch(
                current_batch, process_function, batch_number, optimal_workers
            )
            
            # Store results and metrics
            all_results.extend(batch_results)
            
            with self.lock:
                self.batch_metrics.append(batch_metrics)
                
                # Track failed companies for retry
                for result in batch_results:
                    if result.get('status') != 'success':
                        self.failed_companies.append(result.get('company_id', 'unknown'))
            
            # Progress callback
            if progress_callback:
                progress = {
                    'batch_number': batch_number,
                    'companies_processed': len(all_results),
                    'total_companies': len(companies),
                    'current_success_rate': batch_metrics.success_rate,
                    'overall_success_rate': self._calculate_overall_success_rate()
                }
                progress_callback(progress)
            
            # Log batch completion
            logger.info(f"Batch {batch_number} complete: {batch_metrics.successful_count}/{batch_metrics.companies_count} successful")
            logger.info(f"Processing time: {batch_metrics.processing_time:.2f}s, Throughput: {batch_metrics.throughput:.2f} companies/sec")
            
            # Brief pause between batches
            if remaining_companies:
                time.sleep(0.5)
        
        # Retry failed companies if configured
        if self.config.retry_failed_companies and self.failed_companies:
            logger.info(f"\nRetrying {len(self.failed_companies)} failed companies")
            retry_results = self._retry_failed_companies(process_function)
            all_results.extend(retry_results)
        
        return all_results
    
    def _process_single_batch(self, 
                            companies: List[str], 
                            process_function: Callable[[str], Dict[str, Any]],
                            batch_number: int,
                            max_workers: int) -> tuple[List[Dict[str, Any]], BatchMetrics]:
        """Process a single batch with monitoring"""
        start_time = time.time()
        results = []
        successful_count = 0
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all companies
            future_to_company = {
                executor.submit(self._safe_process_company, process_function, company): company
                for company in companies
            }
            
            # Collect results
            for future in as_completed(future_to_company, timeout=self.config.timeout_per_company * len(companies)):
                company_id = future_to_company[future]
                
                try:
                    result = future.result(timeout=10)  # Additional timeout per company
                    results.append(result)
                    
                    if result.get('status') == 'success':
                        successful_count += 1
                        logger.debug(f"✓ {company_id} processed successfully")
                    else:
                        logger.warning(f"✗ {company_id} processing failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"✗ {company_id} processing exception: {e}")
                    results.append({
                        'company_id': company_id,
                        'status': 'failed',
                        'error': f'Processing exception: {str(e)}'
                    })
        
        # Calculate metrics
        processing_time = time.time() - start_time
        average_time = processing_time / len(companies) if companies else 0
        throughput = len(companies) / processing_time if processing_time > 0 else 0
        
        metrics = BatchMetrics(
            batch_number=batch_number,
            companies_count=len(companies),
            successful_count=successful_count,
            failed_count=len(companies) - successful_count,
            processing_time=processing_time,
            average_time_per_company=average_time,
            throughput=throughput
        )
        
        return results, metrics
    
    def _safe_process_company(self, process_function: Callable, company_id: str) -> Dict[str, Any]:
        """Safely process a single company with timeout protection"""
        try:
            result = process_function(company_id)
            return result
        except Exception as e:
            return {
                'company_id': company_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _retry_failed_companies(self, process_function: Callable) -> List[Dict[str, Any]]:
        """Retry failed companies with reduced batch size"""
        if not self.failed_companies:
            return []
        
        logger.info(f"Retrying {len(self.failed_companies)} failed companies")
        
        # Use smaller batches for retry
        retry_config = BatchConfig(
            batch_size=min(3, self.config.batch_size // 2),
            max_workers=max(1, self.config.max_workers // 2),
            timeout_per_company=self.config.timeout_per_company * 2,  # More time for retries
            retry_failed_companies=False,  # Avoid infinite retry
            adaptive_sizing=False
        )
        
        retry_processor = EnhancedBatchProcessor(retry_config)
        retry_results = retry_processor.process_companies_optimized(
            self.failed_companies.copy(), 
            process_function
        )
        
        # Clear failed companies list
        self.failed_companies.clear()
        
        return retry_results
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all batches"""
        if not self.batch_metrics:
            return 0.0
        
        total_companies = sum(m.companies_count for m in self.batch_metrics)
        total_successful = sum(m.successful_count for m in self.batch_metrics)
        
        return (total_successful / max(total_companies, 1)) * 100
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        if not self.batch_metrics:
            return {'status': 'no_data'}
        
        # Aggregate metrics
        total_companies = sum(m.companies_count for m in self.batch_metrics)
        total_successful = sum(m.successful_count for m in self.batch_metrics)
        total_failed = sum(m.failed_count for m in self.batch_metrics)
        total_time = sum(m.processing_time for m in self.batch_metrics)
        
        # Calculate averages
        avg_batch_time = statistics.mean([m.processing_time for m in self.batch_metrics])
        avg_throughput = statistics.mean([m.throughput for m in self.batch_metrics])
        avg_success_rate = statistics.mean([m.success_rate for m in self.batch_metrics])
        
        # Batch size evolution
        batch_sizes = [m.companies_count for m in self.batch_metrics]
        
        return {
            'total_statistics': {
                'total_companies': total_companies,
                'successful_companies': total_successful,
                'failed_companies': total_failed,
                'overall_success_rate': (total_successful / max(total_companies, 1)) * 100,
                'total_processing_time': total_time
            },
            'batch_statistics': {
                'total_batches': len(self.batch_metrics),
                'average_batch_time': avg_batch_time,
                'average_throughput': avg_throughput,
                'average_success_rate': avg_success_rate
            },
            'optimization_data': {
                'batch_sizes_used': batch_sizes,
                'min_batch_size': min(batch_sizes) if batch_sizes else 0,
                'max_batch_size': max(batch_sizes) if batch_sizes else 0,
                'adaptive_sizing_enabled': self.config.adaptive_sizing
            },
            'individual_batch_metrics': [
                {
                    'batch_number': m.batch_number,
                    'companies': m.companies_count,
                    'successful': m.successful_count,
                    'success_rate': m.success_rate,
                    'processing_time': m.processing_time,
                    'throughput': m.throughput
                }
                for m in self.batch_metrics
            ]
        }

# Factory function
def create_batch_processor(batch_size: int = 10, max_workers: int = 3, 
                        adaptive: bool = True) -> EnhancedBatchProcessor:
    """Create an optimized batch processor"""
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        adaptive_sizing=adaptive,
        retry_failed_companies=True
    )
    return EnhancedBatchProcessor(config)

# Example usage
if __name__ == "__main__":
    # Example of using the batch processor
    def mock_process_function(company_id: str) -> Dict[str, Any]:
        import random
        time.sleep(random.uniform(0.5, 2.0))  # Simulate processing time
        
        # Simulate some failures
        if random.random() < 0.1:  # 10% failure rate
            return {'company_id': company_id, 'status': 'failed', 'error': 'Mock error'}
        
        return {'company_id': company_id, 'status': 'success', 'score': random.uniform(60, 90)}
    
    # Create processor
    processor = create_batch_processor(batch_size=5, max_workers=2)
    
    # Test companies
    test_companies = [f'COMPANY_{i:03d}' for i in range(1, 21)]  # 20 companies
    
    # Process with progress tracking
    def progress_callback(progress):
        print(f"Batch {progress['batch_number']}: {progress['companies_processed']}/{progress['total_companies']} companies processed")
    
    results = processor.process_companies_optimized(
        test_companies, 
        mock_process_function,
        progress_callback
    )
    
    # Print summary
    summary = processor.get_processing_summary()
    print(f"\nProcessing Summary:")
    print(f"Total companies: {summary['total_statistics']['total_companies']}")
    print(f"Success rate: {summary['total_statistics']['overall_success_rate']:.1f}%")
    print(f"Total time: {summary['total_statistics']['total_processing_time']:.2f}s")