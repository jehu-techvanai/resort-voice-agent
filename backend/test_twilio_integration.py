"""
test_twilio_integration.py — Test script for Twilio/LiveKit integration.

Run with: pytest test_twilio_integration.py -v
Or: python test_twilio_integration.py
"""

import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, '.')


class TestTwilioRouter(unittest.TestCase):
    """Test Twilio webhook handler."""

    def test_generate_twiml_structure(self):
        """Test that TwiML is generated correctly."""
        from twilio_router import generate_twiml

        twiml = generate_twiml(
            stream_url="wss://example.com/ws/twilio",
            stream_token="test-token",
            caller_phone="+1234567890",
            called_phone="+0987654321",
            use_media_stream=True,
        )

        # Check basic TwiML structure
        self.assertIn("<?xml", twiml)
        self.assertIn("<Response>", twiml)
        self.assertIn("<Connect>", twiml)
        # Stream tag may be self-closing
        self.assertTrue("<Stream " in twiml or "<Stream/>" in twiml)
        self.assertIn("wss://example.com/ws/twilio", twiml)

    def test_generate_twiml_fallback_mode(self):
        """Test TwiML generation in fallback mode."""
        from twilio_router import generate_twiml

        twiml = generate_twiml(
            stream_url="wss://example.com/ws/twilio",
            stream_token="test-token",
            caller_phone="+1234567890",
            called_phone="+0987654321",
            use_media_stream=False,
        )

        # Check fallback structure (TwiML uses capitalized tags)
        self.assertIn("<Gather ", twiml)
        self.assertIn("speech", twiml)
        self.assertIn("Hello, I'm listening", twiml)


class TestLiveKitWorker(unittest.TestCase):
    """Test LiveKit worker functions."""

    def test_extract_phone_from_room_name(self):
        """Test phone extraction from room names."""
        # Skip if livekit not installed
        try:
            from livekit_worker import extract_phone_from_room_name
        except ModuleNotFoundError:
            self.skipTest("LiveKit not installed")

        # Test SIP URI format
        self.assertEqual(
            extract_phone_from_room_name("sip:+1234567890@sip.livekit.cloud"),
            "+1234567890"
        )

        # Test plain phone number
        self.assertEqual(
            extract_phone_from_room_name("1234567890"),
            "1234567890"
        )

        # Test with extension
        self.assertEqual(
            extract_phone_from_room_name("sip:+1234567890*123@sip.livekit.cloud"),
            "+1234567890"
        )

        # Test invalid input
        self.assertIsNone(extract_phone_from_room_name(""))
        self.assertIsNone(extract_phone_from_room_name("invalid-room"))


class TestConfigIntegration(unittest.TestCase):
    """Test config integration."""

    def test_twilio_config_reads_from_env(self):
        """Test that Twilio config reads from environment."""
        import os
        from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER

        # These should be strings (possibly empty if not set)
        self.assertIsInstance(TWILIO_ACCOUNT_SID, str)
        self.assertIsInstance(TWILIO_AUTH_TOKEN, str)
        self.assertIsInstance(TWILIO_PHONE_NUMBER, str)

    def test_livekit_config_reads_from_env(self):
        """Test that LiveKit config reads from environment."""
        from config import (
            LIVEKIT_URL,
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET,
            LIVEKIT_SIP_URI,
            SIP_OUTBOUND_TRUNK_ID,
        )

        # These should be strings (possibly empty if not set)
        self.assertIsInstance(LIVEKIT_URL, str)
        self.assertIsInstance(LIVEKIT_API_KEY, str)
        self.assertIsInstance(LIVEKIT_API_SECRET, str)


class TestTwilioSignatureValidation(unittest.TestCase):
    """Test Twilio signature validation."""

    def test_validate_twilio_signature_function_exists(self):
        """Test that signature validation function exists."""
        from twilio_router import validate_twilio_signature

        # Test with dummy data
        result = validate_twilio_signature(
            url="https://example.com/twilio/inbound",
            params={"CallSid": "test"},
            signature="invalid-signature",
            auth_token="test-token",
        )

        # Should return False for invalid signature
        self.assertFalse(result)


class TestIntegrationEndpoints(unittest.IsolatedAsyncioTestCase):
    """Test integration endpoints."""

    async def test_health_endpoint(self):
        """Test that health endpoint works."""
        from fastapi.testclient import TestClient
        from app import app

        client = TestClient(app)
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ok", data)

    async def test_twilio_inbound_missing_credentials(self):
        """Test Twilio inbound when credentials are missing."""
        from fastapi.testclient import TestClient
        from app import app

        client = TestClient(app)

        # Mock form data
        form_data = {
            "CallSid": "test-call-sid",
            "From": "+1234567890",
            "To": "+0987654321",
            "CallStatus": "queued",
            "Direction": "inbound",
        }

        # Without credentials configured, should return fallback response
        with patch.dict('os.environ', {}, clear=True):
            from config import TWILIO_ACCOUNT_SID
            # Force reimport
            import importlib
            import config
            importlib.reload(config)

            response = client.post(
                "/twilio/inbound",
                data=form_data,
            )

            # Should return a response (fallback or error)
            self.assertIn(response.status_code, [200, 400, 403])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
