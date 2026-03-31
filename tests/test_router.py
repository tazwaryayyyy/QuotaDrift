"""
Integration tests for QuotaDrift router and provider failover.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
import router as ai_router
from model_manager import model_manager


class TestRouterFailover:
    """Test router failover behavior with mocked providers."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router with controlled responses."""
        with patch('router.get_router') as mock_get_router:
            mock_router_instance = AsyncMock()
            mock_get_router.return_value = mock_router_instance
            yield mock_router_instance

    @pytest.mark.asyncio
    async def test_primary_provider_success(self, mock_router):
        """Test successful response from primary provider (Groq)."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.model = "groq/llama-3.3-70b-versatile"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 10

        mock_router.acompletion.return_value = mock_response

        # Test chat
        result = await ai_router.chat([{"role": "user", "content": "test"}])

        assert result["content"] == "Test response"
        assert result["model_used"] == "groq/llama-3.3-70b-versatile"
        assert result["tokens"] == 10

    @pytest.mark.asyncio
    async def test_failover_chain(self, mock_router):
        """Test failover through the provider chain."""
        # Mock rate limit error for primary, success for secondary
        rate_limit_error = Exception("Rate limit exceeded")
        success_response = MagicMock()
        success_response.model = "openai/gpt-4o-mini"
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Fallback response"
        success_response.usage.total_tokens = 15

        # First call fails, second succeeds
        mock_router.acompletion.side_effect = [
            rate_limit_error,
            success_response
        ]

        # Test chat with retry logic
        with patch('router._try_mark_error') as mock_mark_error:
            result = await ai_router.chat([{"role": "user", "content": "test"}])

            # Should have retried and succeeded
            assert result["content"] == "Fallback response"
            assert result["model_used"] == "openai/gpt-4o-mini"
            mock_mark_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_router):
        """Test streaming chat response."""
        # Mock streaming chunks
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
        ]

        chunks[0].model = "groq/llama-3.3-70b-versatile"

        mock_async_iter = AsyncMock()
        mock_async_iter.__aiter__ = AsyncMock(return_value=iter(chunks))
        mock_router.acompletion.return_value = mock_async_iter

        # Collect streaming response
        events = []
        async for event in ai_router.stream_chat([{"role": "user", "content": "test"}]):
            events.append(event)

        # Should have token events and done event
        assert len(events) == 4  # 3 tokens + 1 done
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "Hello"
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_provider_timeout_handling(self, mock_router):
        """Test timeout handling for slow providers."""
        # Mock timeout error
        mock_router.acompletion.side_effect = TimeoutError("Request timeout")

        with pytest.raises(TimeoutError):
            await ai_router.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with router."""
        # Test circuit breaker state changes
        circuit_breaker = model_manager.model_manager.circuit_breakers["primary"]

        # Simulate failures
        for i in range(5):  # Exceed failure threshold
            model_manager.model_manager.record_failure("primary", f"req_{i}", "Test error")

        # Should be open now
        assert circuit_breaker.state == "open"

        # Try to get best model - should skip failed provider
        best_model = model_manager.model_manager.get_best_model("test_req")
        assert best_model != "primary"

class TestProviderTestEndpoint:
    """Test the /api/provider-test endpoint."""

    @pytest.mark.asyncio
    async def test_provider_test_endpoint(self):
        """Test provider test endpoint with mocked responses."""
        import os

        from main import test_providers

        # Mock environment variables
        with patch.dict(os.environ, {
            'GROQ_API_KEY': 'test_key',
            'MISTRAL_API_KEY': 'test_key',
            'SILICONFLOW_API_KEY': 'test_key',
            'HUGGINGFACE_API_KEY': 'test_key',
            'CLOUDFLARE_API_KEY': 'test_key',
            'CLOUDFLARE_ACCOUNT_ID': 'test_account',
            'OPENROUTER_API_KEY': 'test_key'
        }):
            # Mock router responses
            with patch('main.ai_router.chat') as mock_chat:
                mock_chat.return_value = {
                    "content": "test",
                    "model_used": "groq/llama-3.3-70b-versatile",
                    "tokens": 5
                }

                result = await test_providers()

                assert "results" in result
                assert "summary" in result
                assert result["summary"]["total_providers"] == 7  # All providers checked
                assert result["results"]["groq"]["status"] == "success"
                assert result["results"]["mistral"]["status"] == "success"

class TestModelManagerIntegration:
    """Test model manager integration with router."""

    @pytest.mark.asyncio
    async def test_request_tracking(self):
        """Test request tracking through model manager."""
        request_id = model_manager.model_manager.get_request_id()

        # Get best model (should be primary if available)
        model_name = model_manager.model_manager.get_best_model(request_id)
        assert model_name in config.health

        # Record success
        model_manager.model_manager.record_success(model_name, request_id, tokens=10)

        # Check metrics updated
        metrics = model_manager.model_manager.metrics[model_name]
        assert len(metrics.recent_successes) > 0
        assert metrics.total_requests > 0

    @pytest.mark.asyncio
    async def test_dynamic_scoring(self):
        """Test dynamic model scoring based on performance."""
        # Simulate some successful requests
        for _i in range(5):
            request_id = model_manager.model_manager.get_request_id()
            model_name = "primary"

            # Record with low latency (good performance)
            model_manager.model_manager.start_request(request_id, model_name)
            await asyncio.sleep(0.01)  # Simulate processing
            model_manager.model_manager.record_success(model_name, request_id, tokens=5)

        # Update metrics
        model_manager.model_manager._update_metrics("primary")

        # Check score is calculated
        metrics = model_manager.model_manager.metrics["primary"]
        assert metrics.score > 0
        assert metrics.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_health_snapshot(self):
        """Test health snapshot includes all new providers."""
        snapshot = model_manager.model_manager.get_health_snapshot()

        # Should include all configured providers
        provider_names = [item["slot"] for item in snapshot]
        expected_providers = ["primary", "secondary", "tertiary", "quaternary", "siliconflow", "huggingface", "cloudflare", "fallback"]

        for provider in expected_providers:
            assert provider in provider_names

        # Check health data structure
        for item in snapshot:
            assert "score" in item
            assert "success_rate" in item
            assert "circuit_state" in item

class TestStreamingEvents:
    """Test streaming event format and SSE compliance."""

    @pytest.mark.asyncio
    async def test_sse_event_format(self):
        """Test Server-Sent Events format compliance."""
        # Mock streaming response
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
        ]

        chunks[0].model = "groq/llama-3.3-70b-versatile"

        mock_async_iter = AsyncMock()
        mock_async_iter.__aiter__ = AsyncMock(return_value=iter(chunks))

        with patch('router.get_router') as mock_get_router:
            mock_router_instance = AsyncMock()
            mock_router_instance.acompletion.return_value = mock_async_iter
            mock_get_router.return_value = mock_router_instance

            # Collect events
            events = []
            async for event in ai_router.stream_chat([{"role": "user", "content": "test"}]):
                events.append(event)

            # Verify event structure
            for event in events[:-1]:  # All except final 'done' event
                assert "type" in event
                assert "content" in event
                assert event["type"] == "token"

            # Final event should be 'done'
            assert events[-1]["type"] == "done"
            assert "model" in events[-1]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
