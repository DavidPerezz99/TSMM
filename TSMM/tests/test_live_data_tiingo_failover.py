import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.live_data import bootstrap_master_on_backend_start, update_fx_master_table_db


class LiveDataTiingoFailoverTests(unittest.TestCase):
    @patch("utils.live_data.upsert_ohlc_1m", return_value=1)
    @patch("utils.live_data.get_latest_date", return_value="2026-06-04 10:00:00")
    @patch("utils.live_data.init_market_db")
    @patch("utils.live_data.requests.get")
    def test_db_update_rotates_token_after_quota(self, get_mock, _init_db_mock, _latest_mock, _upsert_mock):
        quota_resp = Mock()
        quota_resp.status_code = 429
        quota_resp.text = "API quota exceeded"

        ok_resp = Mock()
        ok_resp.status_code = 200
        ok_resp.text = "ok"
        ok_resp.json.return_value = [
            {
                "date": "2026-06-04T10:01:00Z",
                "open": 4500.0,
                "high": 4501.0,
                "low": 4499.0,
                "close": 4500.5,
            }
        ]

        get_mock.side_effect = [quota_resp, ok_resp]

        with TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "tiingo_rotation_state.json")
            with patch.dict(
                os.environ,
                {
                    "TIINGO_API_TOKEN": "token_primary",
                    "TIINGO_API_TOKEN_ALT": "token_alt",
                },
                clear=False,
            ):
                out = update_fx_master_table_db(
                    db_path=os.path.join(tmpdir, "market.sqlite"),
                    rate="1min",
                    symbol="xauusd",
                    token_env="TIINGO_API_TOKEN",
                    token_envs=["TIINGO_API_TOKEN", "TIINGO_API_TOKEN_ALT"],
                    token_rotation_state_path=state_path,
                )

            self.assertTrue(out.get("updated"))
            self.assertTrue(out.get("token_rotated"))
            self.assertEqual(out.get("used_token_env"), "TIINGO_API_TOKEN_ALT")
            self.assertEqual(get_mock.call_count, 2)
            self.assertIn("token=token_primary", get_mock.call_args_list[0].args[0])
            self.assertIn("token=token_alt", get_mock.call_args_list[1].args[0])

            with open(state_path, "r", encoding="utf-8") as f:
                state_payload = json.load(f)
            self.assertEqual(state_payload.get("current_env"), "TIINGO_API_TOKEN_ALT")

    @patch("utils.live_data.upsert_ohlc_1m", return_value=1)
    @patch("utils.live_data.get_latest_date", return_value="2026-06-04 10:00:00")
    @patch("utils.live_data.init_market_db")
    @patch("utils.live_data.requests.get")
    def test_db_update_uses_persisted_active_token_first(self, get_mock, _init_db_mock, _latest_mock, _upsert_mock):
        ok_resp = Mock()
        ok_resp.status_code = 200
        ok_resp.text = "ok"
        ok_resp.json.return_value = [
            {
                "date": "2026-06-04T10:01:00Z",
                "open": 4500.0,
                "high": 4501.0,
                "low": 4499.0,
                "close": 4500.5,
            }
        ]
        get_mock.return_value = ok_resp

        with TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "tiingo_rotation_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"current_env": "TIINGO_API_TOKEN_ALT"}, f)

            with patch.dict(
                os.environ,
                {
                    "TIINGO_API_TOKEN": "token_primary",
                    "TIINGO_API_TOKEN_ALT": "token_alt",
                },
                clear=False,
            ):
                out = update_fx_master_table_db(
                    db_path=os.path.join(tmpdir, "market.sqlite"),
                    rate="1min",
                    symbol="xauusd",
                    token_env="TIINGO_API_TOKEN",
                    token_envs=["TIINGO_API_TOKEN", "TIINGO_API_TOKEN_ALT"],
                    token_rotation_state_path=state_path,
                )

            self.assertTrue(out.get("updated"))
            self.assertEqual(get_mock.call_count, 1)
            self.assertIn("token=token_alt", get_mock.call_args.args[0])
            self.assertEqual(out.get("used_token_env"), "TIINGO_API_TOKEN_ALT")

    def test_bootstrap_reports_missing_token_pool(self):
        with patch.dict(
            os.environ,
            {
                "TIINGO_API_TOKEN": "",
                "TIINGO_API_TOKEN_ALT": "",
            },
            clear=False,
        ):
            out = bootstrap_master_on_backend_start(
                master_table_path="data/market_data.sqlite",
                rate="1min",
                symbol="xauusd",
                token_env="TIINGO_API_TOKEN",
                token_envs=["TIINGO_API_TOKEN", "TIINGO_API_TOKEN_ALT"],
            )

        self.assertFalse(out.get("ok"))
        self.assertIn("Missing Tiingo token in configured env vars", str(out.get("error")))


if __name__ == "__main__":
    unittest.main()
