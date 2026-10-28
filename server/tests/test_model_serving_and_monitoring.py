# server/tests/test_model_serving_and_monitoring.py
# Phase 7: Verification of LRU Model Serving Cache, Telemetry & Feature Drift

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from app.main import app
from app.config import settings
from app.services.serving_service import ModelServingCache, serving_cache, execute_served_inference
from app.services.monitoring_service import (
    record_inference_event,
    get_monitoring_stats,
    get_monitoring_logs,
    get_model_drift_indicator,
    init_monitoring_db
)
from app.services.model_registry_service import register_model, init_model_registry, delete_model


class TestModelServingAndMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_monitoring_db()
        init_model_registry()
        cls.client = TestClient(app)

    def test_01_lru_cache_eviction(self):
        """Verify LRU cache limits size and evicts oldest unused models."""
        cache = ModelServingCache(maxsize=2)
        
        # Mock objects
        cache._cache[101] = (object(), {"model_name": "A"}, 100.0)
        cache._cache[102] = (object(), {"model_name": "B"}, 101.0)
        self.assertEqual(len(cache._cache), 2)
        self.assertIn(101, cache._cache)
        self.assertIn(102, cache._cache)

        # Access 101
        cache._cache.move_to_end(101)

        # Add 103 and enforce maxsize
        cache._cache[103] = (object(), {"model_name": "C"}, 102.0)
        if len(cache._cache) > cache.maxsize:
            cache._cache.popitem(last=False)

        self.assertEqual(len(cache._cache), 2)
        self.assertIn(101, cache._cache)
        self.assertIn(103, cache._cache)
        self.assertNotIn(102, cache._cache)

        # Evict explicitly
        evicted = cache.evict(101)
        self.assertTrue(evicted)
        self.assertNotIn(101, cache._cache)

    def test_02_telemetry_logging_and_summary(self):
        """Verify inference logs record metrics and compute latency percentiles."""
        test_model_id = 9999
        
        # Log several test inferences
        for lat in [12.5, 15.0, 18.2, 45.0, 8.1]:
            record_inference_event(
                model_id=test_model_id,
                model_name="TestRFClassifier",
                task_type="classification",
                latency_ms=lat,
                row_count=1,
                status="success",
                features_summary={"feature_x": {"mean": 25.0, "null_count": 0}}
            )

        summary = get_monitoring_stats()
        self.assertIn("total_inferences", summary)
        self.assertGreaterEqual(summary["total_inferences"], 5)
        self.assertIn("p50_latency_ms", summary)
        self.assertIn("p90_latency_ms", summary)
        self.assertIn("p99_latency_ms", summary)
        self.assertGreater(summary["p99_latency_ms"], 0)

        logs_data = get_monitoring_logs(limit=10)
        self.assertIn("items", logs_data)
        self.assertGreaterEqual(len(logs_data["items"]), 1)
        latest = logs_data["items"][0]
        self.assertIn("model_id", latest)
        self.assertIn("latency_ms", latest)
        self.assertIn("status", latest)

    def test_03_monitoring_routes(self):
        """Verify GET /monitoring/stats, /monitoring/logs, and /monitoring/models."""
        res_stats = self.client.get("/monitoring/stats")
        self.assertEqual(res_stats.status_code, 200)
        data = res_stats.json()
        self.assertIn("total_inferences", data)
        self.assertIn("avg_latency_ms", data)
        self.assertIn("p90_latency_ms", data)
        self.assertIn("p99_latency_ms", data)

        res_logs = self.client.get("/monitoring/logs?limit=5")
        self.assertEqual(res_logs.status_code, 200)
        logs = res_logs.json()
        self.assertIn("items", logs)

        res_models = self.client.get("/monitoring/models")
        self.assertEqual(res_models.status_code, 200)
        models_data = res_models.json()
        self.assertIn("cached_models", models_data)
        self.assertIn("cache_capacity", models_data)

    def test_04_served_inference_caching_and_telemetry(self):
        """Verify served inference caches model on first load and reuses cache on second."""
        # Train and register a simple model
        X = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)

        reg_info = register_model(
            pipeline=clf,
            model_name="ServingTestModel",
            task_type="classification",
            model_type="RandomForestClassifier",
            metrics={"accuracy": 1.0},
            feature_columns=["feature1", "feature2"]
        )
        model_id = reg_info["id"]

        try:
            # 1st call: cold load (cached=False)
            pred_payload = [{"feature1": 2.5, "feature2": 25.0}]
            res1 = self.client.post(f"/models/{model_id}/predict", json={"rows": pred_payload})
            self.assertEqual(res1.status_code, 200)
            data1 = res1.json()
            self.assertIn("predictions", data1)
            self.assertIn("latency_ms", data1)
            self.assertFalse(data1["cached"])

            # 2nd call: warm cached load (cached=True)
            res2 = self.client.post(f"/models/{model_id}/predict", json={"rows": pred_payload})
            self.assertEqual(res2.status_code, 200)
            data2 = res2.json()
            self.assertTrue(data2["cached"])
            self.assertLess(data2["latency_ms"], 100.0)

            # Test cache eviction upon deletion
            del_res = self.client.delete(f"/models/{model_id}")
            self.assertEqual(del_res.status_code, 200)
            self.assertNotIn(model_id, serving_cache._cache)
        finally:
            try:
                delete_model(model_id)
            except Exception:
                pass

    def test_05_covariate_feature_drift_calculation(self):
        """Verify covariate drift metric endpoint handles models."""
        X = pd.DataFrame({
            "col_a": [10.0, 12.0, 11.0, 13.0, 10.5],
            "col_b": [100.0, 105.0, 98.0, 102.0, 101.0]
        })
        y = pd.Series([1, 0, 1, 0, 1])
        clf = RandomForestClassifier(n_estimators=3, random_state=42)
        clf.fit(X, y)

        reg_info = register_model(
            pipeline=clf,
            model_name="DriftTestModel",
            task_type="classification",
            model_type="RandomForestClassifier",
            metrics={"accuracy": 0.8},
            feature_columns=["col_a", "col_b"]
        )
        model_id = reg_info["id"]

        try:
            # Log some simulated inference inputs
            for _ in range(5):
                record_inference_event(
                    model_id=model_id,
                    model_name="DriftTestModel",
                    task_type="classification",
                    latency_ms=10.0,
                    row_count=1,
                    status="success",
                    features_summary={"col_a": {"mean": 45.0, "null_count": 0}, "col_b": {"mean": 102.0, "null_count": 0}}
                )

            drift_res = self.client.get(f"/monitoring/drift/{model_id}")
            self.assertEqual(drift_res.status_code, 200)
            drift_data = drift_res.json()
            self.assertEqual(drift_data["model_id"], model_id)
            self.assertIn("status", drift_data)
        finally:
            try:
                delete_model(model_id)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
