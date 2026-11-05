"""Tests for eagleplot/nist.py — NIST WebBook integration."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.nist import (
    get_compound_url,
    get_ir_jcamp_url,
    search_url,
    fetch_ir_spectrum,
    list_ir_spectra,
)

# Fake DataFrame returned by the mocked load_jcamp
_FAKE_DF = pd.DataFrame({"x": [400.0, 1000.0, 2000.0, 4000.0], "y": [0.1, 0.3, 0.5, 0.7]})


class TestURLHelpers:
    def test_get_compound_url(self):
        url = get_compound_url("64-17-5")
        assert "64-17-5" in url
        assert "webbook.nist.gov" in url

    def test_get_ir_jcamp_url(self):
        url = get_ir_jcamp_url("64-17-5", index=0)
        assert "IR-SPEC" in url
        assert "JCAMP=C64175" in url
        assert "Index=0" in url

    def test_get_ir_jcamp_url_index(self):
        url = get_ir_jcamp_url("64-17-5", index=2)
        assert "Index=2" in url

    def test_search_url_name(self):
        url = search_url("ethanol", by="name")
        assert "Name=ethanol" in url

    def test_search_url_formula(self):
        url = search_url("C2H6O", by="formula")
        assert "Formula=C2H6O" in url

    def test_search_url_cas(self):
        url = search_url("64-17-5", by="cas")
        assert "64-17-5" in url

    def test_search_url_name_with_space(self):
        url = search_url("acetic acid", by="name")
        assert "acetic" in url

    def test_get_compound_url_contains_base(self):
        url = get_compound_url("64-17-5")
        assert url.startswith("https://webbook.nist.gov")

    def test_get_ir_jcamp_url_contains_id(self):
        url = get_ir_jcamp_url("64-17-5")
        assert "JCAMP=C64175" in url

    def test_search_url_default_is_name(self):
        url = search_url("water")
        assert "Name=water" in url

    def test_search_url_formula_type(self):
        url = search_url("H2O", by="formula")
        assert "Formula=H2O" in url
        assert "cST=on" in url


_FAKE_JCAMP = """\
##TITLE=Test Compound
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XFACTOR=1.0
##YFACTOR=1.0
##FIRSTX=400
##LASTX=4000
##NPOINTS=4
##XYDATA=(X++(Y..Y))
400 0.1 0.2
1000 0.3 0.4
2000 0.5 0.6
4000 0.7 0.8
##END=
"""


class TestFetchIRSpectrum:
    def _mock_response(self, text, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.text = text
        return mock

    @patch("eagleplot.io.load_jcamp", return_value=_FAKE_DF.copy())
    @patch("requests.get")
    def test_basic(self, mock_get, mock_jcamp):
        mock_get.return_value = self._mock_response(_FAKE_JCAMP)
        df = fetch_ir_spectrum("64-17-5")
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) > 0
        assert df.attrs.get("nist_cas") == "64-17-5"

    @patch("eagleplot.io.load_jcamp", return_value=_FAKE_DF.copy())
    @patch("requests.get")
    def test_attrs_set(self, mock_get, mock_jcamp):
        mock_get.return_value = self._mock_response(_FAKE_JCAMP)
        df = fetch_ir_spectrum("64-17-5")
        assert "nist_url" in df.attrs
        assert "webbook.nist.gov" in df.attrs["nist_url"]

    @patch("requests.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = self._mock_response("Not Found", status=404)
        with pytest.raises(RuntimeError, match="HTTP 404"):
            fetch_ir_spectrum("bad-cas")

    @patch("requests.get")
    def test_no_spectrum(self, mock_get):
        mock_get.return_value = self._mock_response("<html>No spectrum</html>")
        with pytest.raises(RuntimeError, match="No IR spectrum found"):
            fetch_ir_spectrum("99-99-9")

    @patch("eagleplot.io.load_jcamp", return_value=_FAKE_DF.copy())
    @patch("requests.get")
    def test_returns_dataframe(self, mock_get, mock_jcamp):
        mock_get.return_value = self._mock_response(_FAKE_JCAMP)
        df = fetch_ir_spectrum("64-17-5")
        assert isinstance(df, pd.DataFrame)

    @patch("eagleplot.io.load_jcamp", return_value=_FAKE_DF.copy())
    @patch("requests.get")
    def test_nist_url_attr_contains_cas(self, mock_get, mock_jcamp):
        mock_get.return_value = self._mock_response(_FAKE_JCAMP)
        df = fetch_ir_spectrum("64-17-5")
        assert "64-17-5" in df.attrs["nist_url"]

    @patch("eagleplot.io.load_jcamp", return_value=_FAKE_DF.copy())
    @patch("requests.get")
    def test_index_passed_to_url(self, mock_get, mock_jcamp):
        mock_get.return_value = self._mock_response(_FAKE_JCAMP)
        fetch_ir_spectrum("64-17-5", index=2)
        called_url = mock_get.call_args[0][0]
        assert "Index=2" in called_url

    def test_no_requests_raises(self, monkeypatch):
        """ImportError is raised when requests is not importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("No module named 'requests'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="requests"):
            fetch_ir_spectrum("64-17-5")


class TestListIRSpectra:
    @patch("requests.get")
    def test_returns_list(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_FAKE_JCAMP)
        results = list_ir_spectra("64-17-5")
        assert isinstance(results, list)
        if results:
            assert "index" in results[0]
            assert "url" in results[0]

    @patch("requests.get")
    def test_stops_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404, text="")
        results = list_ir_spectra("64-17-5")
        assert results == []

    @patch("requests.get")
    def test_stops_on_no_title(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html>not jcamp</html>")
        results = list_ir_spectra("64-17-5")
        assert results == []

    @patch("requests.get")
    def test_result_has_title(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_FAKE_JCAMP)
        results = list_ir_spectra("64-17-5")
        assert len(results) > 0
        assert results[0]["title"] == "Test Compound"

    @patch("requests.get")
    def test_result_has_data_type(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_FAKE_JCAMP)
        results = list_ir_spectra("64-17-5")
        assert len(results) > 0
        assert "data_type" in results[0]

    @patch("requests.get")
    def test_result_url_is_string(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_FAKE_JCAMP)
        results = list_ir_spectra("64-17-5")
        if results:
            assert isinstance(results[0]["url"], str)
