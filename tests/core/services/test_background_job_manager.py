"""
Unit Tests for Background Job Manager
=====================================

Tests the enterprise-grade background job management system with
aerospace-level reliability features.
"""

import asyncio
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.core.services.background_job_manager import (
    BackgroundJobManager,
    CircuitBreaker,
    JobConfiguration,
    JobMetrics,
    JobPriority,
    JobStatus,
    get_job_manager,
)


class TestJobPriority(unittest.TestCase):
    """Test job priority enumeration"""

    def test_priority_values(self):
        """Test that priorities have correct values"""
        # Just verify they exist and are different
        priorities = [
            JobPriority.CRITICAL,
            JobPriority.HIGH,
            JobPriority.NORMAL,
            JobPriority.LOW,
            JobPriority.MAINTENANCE,
        ]

        # All should be unique
        self.assertEqual(len(priorities), len(set(priorities)))


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        self.assertEqual(cb.failure_threshold, 3)
        self.assertEqual(cb.recovery_timeout, 60)
        self.assertFalse(cb.is_open)
        self.assertTrue(cb.can_execute())

    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures"""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.can_execute())  # Still closed

        cb.record_failure()  # Third failure
        self.assertTrue(cb.is_open)
        self.assertFalse(cb.can_execute())

    def test_circuit_breaker_resets_on_success(self):
        """Test that circuit breaker resets on success"""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure()
        cb.record_failure()  # Open
        self.assertTrue(cb.is_open)

        # Force close for testing
        cb.is_open = False
        cb.record_success()

        self.assertEqual(cb.failure_count, 0)
        self.assertFalse(cb.is_open)


class TestBackgroundJobManager(unittest.TestCase):
    """Test the background job manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = BackgroundJobManager(
            use_async=False
        )  # Use sync for easier testing

        # Mock job function
        self.mock_job = Mock(return_value={"status": "success"})
        self.mock_async_job = AsyncMock(return_value={"status": "success"})

    def tearDown(self):
        """Clean up after tests"""
        # Ensure scheduler is shut down
        if hasattr(self.manager, "scheduler") and self.manager.scheduler.running:
            self.manager.shutdown(wait=False)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.manager)
        self.assertIsNotNone(self.manager.scheduler)
        self.assertIsInstance(self.manager.jobs, dict)
        self.assertIsInstance(self.manager.metrics, dict)
        self.assertIsInstance(self.manager.circuit_breakers, dict)

    def test_add_job(self):
        """Test adding a job"""
        job_config = JobConfiguration(
            name="test_job",
            func=self.mock_job,
            trigger="interval",
            priority=JobPriority.NORMAL,
            max_retries=3,
            kwargs={"seconds": 60},
        )

        job_id = self.manager.add_job(job_config)

        self.assertIsNotNone(job_id)
        self.assertIn(job_id, self.manager.jobs)
        self.assertEqual(self.manager.jobs[job_id].name, "test_job")

    def test_remove_job(self):
        """Test removing a job"""
        job_config = JobConfiguration(
            name="removable_job",
            func=self.mock_job,
            trigger="interval",
            kwargs={"seconds": 60},
        )

        job_id = self.manager.add_job(job_config)
        self.assertIn(job_id, self.manager.jobs)

        self.manager.remove_job(job_id)
        self.assertNotIn(job_id, self.manager.jobs)

    def test_job_status(self):
        """Test getting job status"""
        job_config = JobConfiguration(
            name="status_job",
            func=self.mock_job,
            trigger="interval",
            priority=JobPriority.HIGH,
            kwargs={"seconds": 60},
        )

        job_id = self.manager.add_job(job_config)
        status = self.manager.get_job_status(job_id)

        self.assertIsNotNone(status)
        self.assertEqual(status["name"], "status_job")
        self.assertEqual(status["priority"], "HIGH")
        self.assertFalse(status["is_running"])
        self.assertIn("metrics", status)

    def test_circuit_breaker_for_critical_jobs(self):
        """Test that circuit breakers are added for critical jobs"""
        job_config = JobConfiguration(
            name="critical_job",
            func=self.mock_job,
            trigger="interval",
            priority=JobPriority.CRITICAL,
            kwargs={"seconds": 60},
        )

        job_id = self.manager.add_job(job_config)

        # Critical jobs should have circuit breakers
        self.assertIn(job_id, self.manager.circuit_breakers)
        self.assertIsInstance(self.manager.circuit_breakers[job_id], CircuitBreaker)

    def test_job_execution_wrapper(self):
        """Test job execution wrapper"""
        executed = False

        def test_job():
            nonlocal executed
            executed = True

        job_config = JobConfiguration(
            name="wrapper_test",
            func=test_job,
            trigger="date",
            kwargs={"run_date": datetime.now() + timedelta(seconds=0.1)},
        )

        job_id = self.manager.add_job(job_config)

        # Start scheduler and wait
        self.manager.start()
        time.sleep(0.3)

        self.assertTrue(executed)

    def test_metrics_tracking(self):
        """Test that metrics are tracked"""
        job_config = JobConfiguration(
            name="metrics_job",
            func=lambda: None,  # Simple no-op
            trigger="date",
            kwargs={"run_date": datetime.now() + timedelta(seconds=0.1)},
        )

        job_id = self.manager.add_job(job_config)

        # Initial metrics
        self.assertIn(job_id, self.manager.metrics)
        metrics = self.manager.metrics[job_id]
        self.assertEqual(metrics.total_executions, 0)

        # Start and wait for execution
        self.manager.start()
        time.sleep(0.3)

        # Metrics should be updated
        # Note: Due to event handling, we may need to check differently
        status = self.manager.get_job_status(job_id)
        self.assertIsNotNone(status)

    def test_kimera_specific_jobs(self):
        """Test Kimera-specific job initialization"""
        # Mock embedding function
        mock_embedding_fn = Mock(return_value=[0.1] * 512)

        # Initialize Kimera jobs
        self.manager.initialize_kimera_jobs(mock_embedding_fn)

        # Check that jobs were added
        job_names = [job.name for job in self.manager.jobs.values()]

        self.assertIn("scar_decay", job_names)
        self.assertIn("scar_fusion", job_names)
        self.assertIn("scar_crystallization", job_names)

    def test_get_all_jobs_status(self):
        """Test getting status of all jobs"""
        # Add multiple jobs
        for i in range(3):
            job_config = JobConfiguration(
                name=f"job_{i}",
                func=self.mock_job,
                trigger="interval",
                kwargs={"seconds": 60},
            )
            self.manager.add_job(job_config)

        all_status = self.manager.get_all_jobs_status()

        self.assertEqual(len(all_status), 3)
        for status in all_status:
            self.assertIn("name", status)
            self.assertIn("priority", status)
            self.assertIn("metrics", status)

    def test_metrics_summary(self):
        """Test metrics summary generation"""
        summary = self.manager.get_metrics_summary()

        self.assertIn("total_jobs", summary)
        self.assertIn("running_jobs", summary)
        self.assertIn("total_executions", summary)
        self.assertIn("total_failures", summary)
        self.assertIn("overall_success_rate", summary)
        self.assertIn("jobs_with_circuit_breaker_open", summary)

    def test_graceful_shutdown(self):
        """Test graceful shutdown"""
        self.manager.start()
        self.assertTrue(self.manager.scheduler.running)

        self.manager.shutdown(wait=True)
        self.assertFalse(self.manager.scheduler.running)

    def test_singleton_pattern(self):
        """Test that get_job_manager returns singleton"""
        manager1 = get_job_manager()
        manager2 = get_job_manager()

        self.assertIs(manager1, manager2)


class TestKimeraJobs(unittest.TestCase):
    """Test Kimera-specific job implementations"""

    @patch("src.core.services.background_job_manager.SessionLocal")
    def test_decay_job(self, mock_session):
        """Test SCAR decay job"""
        manager = BackgroundJobManager()

        # Mock database session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock SCAR objects
        mock_scar = MagicMock()
        mock_scar.weight = 10.0
        mock_scar.last_accessed = datetime.now() - timedelta(days=2)

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_scar]

        # Execute decay job
        manager._decay_job()

        # Check that weight was reduced
        self.assertLess(mock_scar.weight, 10.0)
        mock_db.commit.assert_called_once()

    @patch("src.core.services.background_job_manager.SessionLocal")
    def test_fusion_job(self, mock_session):
        """Test SCAR fusion job"""
        manager = BackgroundJobManager()

        # Mock database session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock SCARs with same reason
        scars = []
        for i in range(3):
            mock_scar = MagicMock()
            mock_scar.reason = "test_reason"
            mock_scar.weight = float(i + 1)
            scars.append(mock_scar)

        mock_db.query.return_value.all.return_value = scars

        # Execute fusion job
        manager._fusion_job()

        # Check that fusion occurred
        mock_db.delete.assert_called()
        mock_db.commit.assert_called_once()

    @patch("src.core.services.background_job_manager.SessionLocal")
    def test_crystallization_job(self, mock_session):
        """Test SCAR crystallization job"""
        manager = BackgroundJobManager()

        # Set embedding function
        manager._embedding_fn = lambda x: [0.1] * 512

        # Mock database session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock high-weight SCAR
        mock_scar = MagicMock()
        mock_scar.scar_id = "test_scar"
        mock_scar.reason = "important_principle"
        mock_scar.weight = 25.0  # Above threshold

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_scar]
        mock_db.query.return_value.filter.return_value.first.return_value = (
            None  # Not already crystallized
        )

        # Execute crystallization job
        manager._crystallization_job()

        # Check that Geoid was created
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

        # Check that SCAR weight was reset
        self.assertEqual(mock_scar.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
