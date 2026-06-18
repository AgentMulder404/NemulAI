# Copyright 2026 Kevin (NemulAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Backward-compatibility guarantees for the ALUMINATAI_ → NEMULAI_ rename.

The agent's configuration prefix changed from ``ALUMINATAI_`` to ``NEMULAI_``
during the rebrand. Agents and launcher scripts deployed before the change
still set the legacy names, so these must keep working. These tests lock that
contract in place.
"""
import unittest
from unittest import mock

from envcompat import env, env_from, legacy_name, is_set, has_known_prefix
from attribution.process_probe import _filter_environ


class TestEnvFallback(unittest.TestCase):
    def test_reads_current_name(self):
        with mock.patch.dict("os.environ", {"NEMULAI_TEAM": "nlp"}, clear=True):
            self.assertEqual(env("NEMULAI_TEAM"), "nlp")

    def test_falls_back_to_legacy_name(self):
        with mock.patch.dict("os.environ", {"ALUMINATAI_TEAM": "legacy-nlp"}, clear=True):
            self.assertEqual(env("NEMULAI_TEAM"), "legacy-nlp")

    def test_current_name_wins_over_legacy(self):
        env_vars = {"NEMULAI_TEAM": "new", "ALUMINATAI_TEAM": "old"}
        with mock.patch.dict("os.environ", env_vars, clear=True):
            self.assertEqual(env("NEMULAI_TEAM"), "new")

    def test_default_when_neither_set(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(env("NEMULAI_TEAM", "fallback"), "fallback")

    def test_is_set_checks_both_names(self):
        with mock.patch.dict("os.environ", {"ALUMINATAI_API_KEY": "k"}, clear=True):
            self.assertTrue(is_set("NEMULAI_API_KEY"))
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(is_set("NEMULAI_API_KEY"))


class TestEnvFrom(unittest.TestCase):
    """env_from() operates on an arbitrary mapping (e.g. a foreign process env)."""

    def test_legacy_fallback_on_dict(self):
        self.assertEqual(env_from({"ALUMINATAI_MODEL": "llama"}, "NEMULAI_MODEL"), "llama")

    def test_current_wins_on_dict(self):
        d = {"NEMULAI_MODEL": "new", "ALUMINATAI_MODEL": "old"}
        self.assertEqual(env_from(d, "NEMULAI_MODEL"), "new")

    def test_default_on_dict(self):
        self.assertEqual(env_from({}, "NEMULAI_MODEL", "untagged"), "untagged")


class TestPrefixHelpers(unittest.TestCase):
    def test_legacy_name_mapping(self):
        self.assertEqual(legacy_name("NEMULAI_TEAM"), "ALUMINATAI_TEAM")
        self.assertEqual(legacy_name("PATH"), "")

    def test_has_known_prefix(self):
        self.assertTrue(has_known_prefix("NEMULAI_TEAM"))
        self.assertTrue(has_known_prefix("ALUMINATAI_TEAM"))  # legacy still recognized
        self.assertFalse(has_known_prefix("AWS_SECRET_ACCESS_KEY"))


class TestProcessProbeAllowlist(unittest.TestCase):
    """Attribution must retain both prefixes when filtering a process environ."""

    def test_keeps_current_and_legacy_tags(self):
        raw = {
            "NEMULAI_TEAM": "nlp",
            "ALUMINATAI_TEAM": "legacy-nlp",
            "NEMULAI_CUSTOM_TAG": "prod",
            "AWS_SECRET_ACCESS_KEY": "shh",  # must be dropped
        }
        filtered = _filter_environ(raw)
        self.assertIn("NEMULAI_TEAM", filtered)
        self.assertIn("ALUMINATAI_TEAM", filtered)
        self.assertIn("NEMULAI_CUSTOM_TAG", filtered)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", filtered)


if __name__ == "__main__":
    unittest.main()
