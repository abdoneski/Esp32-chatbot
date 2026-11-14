import json
import unittest
from unittest.mock import MagicMock, patch

import llm_fallback
import mac_client
import mac_voice_client


class LlmFallbackTests(unittest.TestCase):
    def setUp(self):
        llm_fallback._api_keys.clear()
        llm_fallback.set_active_provider(None)

    def test_default_provider_selection_without_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(llm_fallback.is_llm_available())
            self.assertIsNone(llm_fallback.get_active_provider())

    def test_gemini_provider_selected_when_key_set(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.assertTrue(llm_fallback.is_llm_available())
            self.assertEqual(llm_fallback.get_active_provider(), "gemini")
            self.assertIn("Gemini", llm_fallback.get_status_description())

    def test_groq_provider_selected_when_key_set(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-groq-key"}, clear=True):
            self.assertTrue(llm_fallback.is_llm_available())
            self.assertEqual(llm_fallback.get_active_provider(), "groq")
            self.assertIn("Groq", llm_fallback.get_status_description())

    def test_in_memory_key_override(self):
        llm_fallback.set_provider_key("gemini", "custom-key")
        self.assertEqual(llm_fallback.get_api_key("gemini"), "custom-key")
        self.assertEqual(llm_fallback.get_active_provider(), "gemini")

    def test_forced_provider_override(self):
        llm_fallback.set_active_provider("off")
        self.assertFalse(llm_fallback.is_llm_available())

    def test_gemini_audio_query(self):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Paris is the capital of France."}]
                    }
                }
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as urlopen:
            result = llm_fallback._query_gemini_audio(b"fake-wav-data", "test-key", 10.0)

        self.assertEqual(result, "Paris is the capital of France.")
        request = urlopen.call_args.args[0]
        self.assertIn("gemini-2.0-flash:generateContent?key=test-key", request.full_url)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["contents"][0]["parts"][0]["inline_data"]["mime_type"], "audio/wav")

    def test_gemini_text_query(self):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "A byte has 8 bits."}]
                    }
                }
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = llm_fallback._query_gemini_text("How many bits in a byte?", "test-key", 10.0)

        self.assertEqual(result, "A byte has 8 bits.")

    def test_groq_audio_query_transcribes_and_answers(self):
        stt_response = MagicMock()
        stt_response.read.return_value = json.dumps({"text": "What is Python?"}).encode("utf-8")
        stt_response.__enter__.return_value = stt_response

        chat_response = MagicMock()
        chat_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Python is a programming language."}}]
        }).encode("utf-8")
        chat_response.__enter__.return_value = chat_response

        with patch("urllib.request.urlopen", side_effect=[stt_response, chat_response]):
            result = llm_fallback._query_groq_audio(b"wav-bytes", "test-key", 10.0)

        self.assertEqual(result, "Python is a programming language.")

    @patch("llm_fallback.is_llm_available", return_value=True)
    @patch("llm_fallback.query_audio", return_value="The capital of France is Paris.")
    @patch.object(mac_voice_client, "send_speech")
    @patch.object(mac_voice_client, "ask_by_voice", return_value="I could not recognize a configured FAQ question.")
    @patch.object(mac_voice_client, "record_audio", return_value=b"wav-data")
    def test_voice_client_falls_back_to_llm_when_faq_unrecognized(
        self, record_audio, ask_by_voice, send_speech, query_audio, is_llm_available
    ):
        with patch("builtins.print"):
            mac_voice_client.send_recording("http://esp32.local", 5.0, 30.0, "esp32")

        query_audio.assert_called_once_with(b"wav-data", timeout=30.0)
        send_speech.assert_called_once_with("http://esp32.local", "The capital of France is Paris.", 30.0)

    @patch("llm_fallback.is_llm_available", return_value=True)
    @patch("llm_fallback.query_text", return_value="The speed of light is 300,000 km/s.")
    @patch.object(mac_client, "ask_esp32", return_value="I don't know that one yet. Please add it to faq.txt.")
    def test_text_client_falls_back_to_llm_when_unknown(
        self, ask_esp32, query_text, is_llm_available
    ):
        with patch("builtins.print") as mock_print:
            mac_client.send_question("http://esp32.local", "What is light speed?", 30.0)

        query_text.assert_called_once_with("What is light speed?", timeout=30.0)
        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertTrue(any("300,000 km/s" in p for p in printed))


if __name__ == "__main__":
    unittest.main()
