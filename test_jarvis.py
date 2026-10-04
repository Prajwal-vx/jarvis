"""
Automated Comprehensive Test Suite for Jarvis Assistant
======================================================
Validates:
- AST Safe Math Evaluator & Security Sandbox (immune to injection)
- Physical Unit Conversion
- Intent & Natural Language Regex Matchers
- Notes Storage Sanitization & Backup Logic
- Offline Wikipedia Knowledge Engine
- System Diagnostics & Resource Monitors
- .env Configuration Loader
"""

import ast
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import main
import client


class TestSafeMathEvaluator(unittest.TestCase):
    """Test the AST math evaluator and its immunity to code injection."""

    def test_basic_arithmetic(self):
        # We test _eval_ast_node directly
        tree = ast.parse("25 * 4", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 100)

        tree = ast.parse("100 / 4", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 25.0)

        tree = ast.parse("2 ** 8", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 256)

        tree = ast.parse("15 + 35 - 10", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 40)

    def test_safe_functions(self):
        tree = ast.parse("sqrt(144)", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 12.0)

        tree = ast.parse("abs(-50)", mode="eval")
        self.assertEqual(main._eval_ast_node(tree), 50)

    def test_division_by_zero(self):
        tree = ast.parse("10 / 0", mode="eval")
        with self.assertRaises(ZeroDivisionError):
            main._eval_ast_node(tree)

    def test_code_injection_security_sandbox(self):
        """Verify that arbitrary code execution via AST is completely blocked."""
        malicious_payloads = [
            "__import__('os').system('echo pwned')",
            "open('pwn.txt', 'w')",
            "eval('1+1')",
            "exec('pass')",
            "[x for x in ().__class__.__bases__[0].__subclasses__()]",
            "import os",
        ]
        for payload in malicious_payloads:
            try:
                tree = ast.parse(payload, mode="eval")
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    main._eval_ast_node(tree)
            except SyntaxError:
                # Statements like 'import os' cannot even be parsed in eval mode
                pass

    def test_expensive_math_is_rejected(self):
        tree = ast.parse("10 ** 100000000", mode="eval")
        with self.assertRaises(ValueError):
            main._eval_ast_node(tree)

        with patch("main.speak") as mock_speak:
            main.calculate_expression("1" * (main.MAX_MATH_EXPRESSION_LENGTH + 1))
        mock_speak.assert_called_once_with("That expression is too long to calculate safely.")

    def test_boolean_and_keyword_arguments_are_rejected(self):
        for expression in ("True", "round(1.2, ndigits=1)"):
            tree = ast.parse(expression, mode="eval")
            with self.assertRaises(ValueError):
                main._validate_math_tree(tree)


class TestDestructiveActionConfirmation(unittest.TestCase):
    @patch("main._run_power_command")
    @patch("main.prompt_user", return_value="yesterday")
    @patch("main.speak")
    def test_substring_yes_does_not_confirm_shutdown(self, _speak, _prompt, run_power):
        main.shutdown_computer()
        run_power.assert_not_called()

    @patch("main._run_power_command")
    @patch("main.prompt_user", return_value=" YES ")
    @patch("main.speak")
    def test_exact_yes_confirms_shutdown(self, _speak, _prompt, run_power):
        main.shutdown_computer()
        run_power.assert_called_once()


class TestUnitConverter(unittest.TestCase):
    """Test unit conversions for temperature, distance, and mass."""

    @patch("main.speak")
    def test_celsius_to_fahrenheit(self, mock_speak):
        res = main.convert_units("convert 100 celsius to fahrenheit")
        self.assertTrue(res)
        mock_speak.assert_called_with("100.0 degrees Celsius is 212.0 degrees Fahrenheit.")

    @patch("main.speak")
    def test_fahrenheit_to_celsius(self, mock_speak):
        res = main.convert_units("convert 32 fahrenheit to celsius")
        self.assertTrue(res)
        mock_speak.assert_called_with("32.0 degrees Fahrenheit is 0.0 degrees Celsius.")

    @patch("main.speak")
    def test_kilometers_to_miles(self, mock_speak):
        res = main.convert_units("convert 10 kilometers to miles")
        self.assertTrue(res)
        mock_speak.assert_called_with("10.0 kilometers is approximately 6.21 miles.")

    @patch("main.speak")
    def test_miles_to_kilometers(self, mock_speak):
        res = main.convert_units("convert 5 miles to km")
        self.assertTrue(res)
        mock_speak.assert_called_with("5.0 miles is approximately 8.05 kilometers.")


class TestNotesManager(unittest.TestCase):
    """Test notes storage sanitization, multi-line safety, and backups."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_notes = os.path.join(self.test_dir, "test_notes.txt")
        self.test_backup = os.path.join(self.test_dir, "test_notes.bak")
        self.orig_notes = main.NOTES_FILE
        self.orig_backup = main.NOTES_BACKUP
        main.NOTES_FILE = self.test_notes
        main.NOTES_BACKUP = self.test_backup

    def tearDown(self):
        main.NOTES_FILE = self.orig_notes
        main.NOTES_BACKUP = self.orig_backup
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("main.speak")
    def test_save_and_read_sanitized_note(self, mock_speak):
        # Multi-line text should be sanitized to a single line
        note_text = "line one\nline two\r\n\t  line three   "
        main.save_note(note_text)

        with open(self.test_notes, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        self.assertIn("line one line two line three", lines[0])

        # Test reading
        main.read_notes()
        mock_speak.assert_any_call("line one line two line three")

    @patch("main.prompt_user", return_value="yes")
    @patch("main.speak")
    def test_clear_notes_with_backup(self, mock_speak, mock_prompt):
        main.save_note("important task")
        self.assertTrue(os.path.exists(self.test_notes))

        main.clear_notes()
        self.assertFalse(os.path.exists(self.test_notes))
        self.assertTrue(os.path.exists(self.test_backup))

    @patch("main.prompt_user", return_value="yesterday")
    @patch("main.speak")
    def test_clear_notes_requires_exact_confirmation(self, _mock_speak, _mock_prompt):
        main.save_note("important task")
        main.clear_notes()

        self.assertTrue(os.path.exists(self.test_notes))
        self.assertFalse(os.path.exists(self.test_backup))


class TestSystemDiagnostics(unittest.TestCase):
    """Test that system monitoring functions execute safely without crashing."""

    @patch("main.speak")
    def test_system_status(self, mock_speak):
        main.system_status()
        self.assertTrue(mock_speak.called)
        spoken_text = mock_speak.call_args[0][0]
        self.assertIn("CPU usage", spoken_text)
        self.assertIn("RAM usage", spoken_text)

    @patch("main.speak")
    def test_system_diagnostics(self, mock_speak):
        main.system_diagnostics()
        self.assertTrue(mock_speak.called)
        spoken_text = mock_speak.call_args[0][0]
        self.assertTrue("storage" in spoken_text or "consumer" in spoken_text or "diagnostics" in spoken_text)


class TestWikipediaKnowledge(unittest.TestCase):
    """Test offline Wikipedia query integration."""

    @patch("main.speak")
    def test_wikipedia_summary(self, mock_speak):
        with patch("main.wikipedia") as wikipedia_mock:
            wikipedia_mock.search.return_value = ["Isaac Newton"]
            wikipedia_mock.summary.return_value = "Isaac Newton was an English physicist."
            res = main.ask_wikipedia("who was Isaac Newton")
        self.assertTrue(res)
        self.assertTrue(mock_speak.called)
        spoken_text = mock_speak.call_args[0][0]
        self.assertIn("According to Wikipedia", spoken_text)
        self.assertIn("Newton", spoken_text)


class TestEnvLoader(unittest.TestCase):
    """Test zero-dependency .env parsing."""

    def test_load_env_file(self):
        temp_env = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
        try:
            temp_env.write("# Comment line\n")
            temp_env.write("TEST_JARVIS_KEY=sample_secret_value\n")
            temp_env.write('TEST_JARVIS_QUOTED="quoted_value"\n')
            temp_env.close()

            client.load_env_file(temp_env.name)
            self.assertEqual(os.environ.get("TEST_JARVIS_KEY"), "sample_secret_value")
            self.assertEqual(os.environ.get("TEST_JARVIS_QUOTED"), "quoted_value")
        finally:
            if os.path.exists(temp_env.name):
                os.remove(temp_env.name)


class TestIntentMatching(unittest.TestCase):
    """Test intent parsing for previously bugged phrases."""

    @patch("main.tell_time")
    def test_time_intents(self, mock_tell_time):
        for phrase in ["what's the time", "what is the time", "tell me the time", "current time", "what time is it"]:
            main.process_command(phrase)
            self.assertTrue(mock_tell_time.called, f"Failed on: {phrase}")
            mock_tell_time.reset_mock()

    @patch("main.tell_date")
    def test_date_intents(self, mock_tell_date):
        for phrase in ["what is the date", "what's the date", "today's date", "current date"]:
            main.process_command(phrase)
            self.assertTrue(mock_tell_date.called, f"Failed on: {phrase}")
            mock_tell_date.reset_mock()

    @patch("main.get_weather")
    def test_weather_intents(self, mock_weather):
        main.process_command("weather in Paris")
        mock_weather.assert_called_with("Paris")
        mock_weather.reset_mock()

        main.process_command("how is the weather like in Tokyo")
        mock_weather.assert_called_with("Tokyo")
        mock_weather.reset_mock()

        main.process_command("London weather")
        mock_weather.assert_called_with("London")


class TestRoutines(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_routines_file = main.ROUTINES_FILE
        main.ROUTINES_FILE = os.path.join(self.test_dir, "routines.json")
        main._active_routines.clear()

    def tearDown(self):
        main.ROUTINES_FILE = self.original_routines_file
        main._active_routines.clear()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("main.speak")
    def test_recursive_routine_is_stopped(self, mock_speak):
        main._save_json(main.ROUTINES_FILE, {"loop": ["run routine loop"]})

        self.assertTrue(main.run_routine("loop"))
        mock_speak.assert_any_call("Routine 'loop' cannot run because it creates a cycle.")
        self.assertFalse(main._active_routines)


class TestConversationContext(unittest.TestCase):
    @patch("client.get_openai_client")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_context_request_sends_complete_history(self, get_client):
        completion = MagicMock()
        completion.choices[0].message.content = "The earlier topic was astronomy."
        client_mock = MagicMock()
        client_mock.chat.completions.create.return_value = completion
        get_client.return_value = client_mock
        history = [
            {"role": "user", "content": "Tell me about astronomy."},
            {"role": "assistant", "content": "Astronomy studies celestial objects."},
            {"role": "user", "content": "What was my earlier topic?"},
        ]

        response = main.ask_openai_with_context(history)

        self.assertEqual(response, "The earlier topic was astronomy.")
        request = client_mock.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][1:], history)


if __name__ == "__main__":
    unittest.main()
