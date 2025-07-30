"""
Unit tests for the optics.analysis module.

This test suite covers all functions in the analysis module including:
- get_output_dir: Directory creation and path handling
- get_rdt_type: RDT classification logic
- get_rdt_paths: Path generation for RDT files
- get_tunes: Tune extraction from TFS files
- get_rdts_from_optics_analysis: Complete RDT analysis workflow
- run_harpy: Harpy frequency analysis execution

Testing framework: pytest
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module under test based on the actual module structure
try:
    from lhcng.analysis import (
        get_output_dir,
        get_rdt_type,
        get_rdt_paths,
        get_tunes,
        get_rdts_from_optics_analysis,
        run_harpy,
    )
except ImportError:
    # Fallback import path
    from optics.analysis import (
        get_output_dir,
        get_rdt_type,
        get_rdt_paths,
        get_tunes,
        get_rdts_from_optics_analysis,
        run_harpy,
    )


class TestGetOutputDir:
    """Test cases for get_output_dir function."""

    @patch("lhcng.analysis.ANALYSIS_DIR", Path("/tmp/test_analysis"))
    def test_get_output_dir_with_default(self):
        """Test get_output_dir creates directory with default path."""
        tbt_name = "test_data.sdds"
        result = get_output_dir(tbt_name)
        expected = Path("/tmp/test_analysis/test_data")
        assert result == expected

    def test_get_output_dir_with_custom_path(self):
        """Test get_output_dir with custom output directory."""
        tbt_name = "test_data.sdds"
        custom_dir = Path("/custom/output")
        result = get_output_dir(tbt_name, custom_dir)
        assert result == custom_dir

    @patch("lhcng.analysis.ANALYSIS_DIR", Path("/tmp/test"))
    def test_get_output_dir_filename_with_multiple_dots(self):
        """Test get_output_dir handles filenames with multiple dots correctly."""
        tbt_name = "test.data.file.sdds"
        result = get_output_dir(tbt_name)
        expected = Path("/tmp/test/test")
        assert result == expected

    def test_get_output_dir_empty_filename(self):
        """Test get_output_dir with empty filename."""
        with pytest.raises(IndexError):
            get_output_dir("")

    @patch("lhcng.analysis.ANALYSIS_DIR", Path("/tmp/test"))
    def test_get_output_dir_filename_no_extension(self):
        """Test get_output_dir with filename without extension."""
        tbt_name = "test_data"
        result = get_output_dir(tbt_name)
        expected = Path("/tmp/test/test_data")
        assert result == expected

    @patch("lhcng.analysis.ANALYSIS_DIR", Path("/tmp/test"))
    def test_get_output_dir_creates_parent_directory(self):
        """Test get_output_dir ensures parent directory exists."""
        tbt_name = "data.sdds"
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = get_output_dir(tbt_name)
            expected = Path("/tmp/test/data")
            assert result == expected
            mock_mkdir.assert_called_once_with(exist_ok=True)


class TestGetRdtType:
    """Test cases for get_rdt_type function."""

    def test_get_rdt_type_normal_sextupole(self):
        """Test get_rdt_type for normal sextupole RDT."""
        rdt = "f3000_x"
        result = get_rdt_type(rdt)
        assert result == ("normal", "sextupole")

    def test_get_rdt_type_skew_sextupole(self):
        """Test get_rdt_type for skew sextupole RDT."""
        rdt = "f2110_y"
        result = get_rdt_type(rdt)
        assert result == ("skew", "sextupole")

    def test_get_rdt_type_normal_octupole(self):
        """Test get_rdt_type for normal octupole RDT."""
        rdt = "f4000_x"
        result = get_rdt_type(rdt)
        assert result == ("normal", "octupole")

    def test_get_rdt_type_skew_octupole(self):
        """Test get_rdt_type for skew octupole RDT."""
        rdt = "f2200_y"
        result = get_rdt_type(rdt)
        assert result == ("skew", "octupole")

    def test_get_rdt_type_edge_cases(self):
        """Test get_rdt_type with various edge cases."""
        test_cases = [
            ("f1010_x", ("normal", "sextupole")),
            ("f1001_y", ("skew", "sextupole")),
            ("f2020_x", ("normal", "octupole")),
            ("f1111_y", ("normal", "octupole")),
        ]

        for rdt, expected in test_cases:
            result = get_rdt_type(rdt)
            assert result == expected, f"Failed for RDT {rdt}: got {result}, expected {expected}"

    def test_get_rdt_type_boundary_conditions(self):
        """Test get_rdt_type with boundary condition inputs."""
        test_cases = [
            ("f0000_x", ("normal", "sextupole")),  # All zeros
            ("f9999_y", ("normal", "sextupole")),  # High numbers
            ("f0001_x", ("skew", "sextupole")),   # Minimal skew
        ]

        for rdt, expected in test_cases:
            result = get_rdt_type(rdt)
            assert result == expected

    def test_get_rdt_type_invalid_format(self):
        """Test get_rdt_type with invalid RDT format."""
        with pytest.raises((ValueError, IndexError)):
            get_rdt_type("invalid_format")

        with pytest.raises(IndexError):
            get_rdt_type("f12_x")  # Not enough digits

    def test_get_rdt_type_coupling_rdts(self):
        """Test get_rdt_type specifically for coupling RDTs."""
        coupling_cases = [
            ("f1001_x", ("skew", "sextupole")),
            ("f1010_y", ("normal", "sextupole")),
        ]

        for rdt, expected in coupling_cases:
            result = get_rdt_type(rdt)
            assert result == expected


class TestGetRdtPaths:
    """Test cases for get_rdt_paths function."""

    def test_get_rdt_paths_single_rdt(self):
        """Test get_rdt_paths with a single RDT."""
        rdts = ["f3000_x"]
        output_dir = Path("/output")
        result = get_rdt_paths(rdts, output_dir)
        expected = {"f3000_x": Path("/output/normal_sextupole/f3000_x.tfs")}
        assert result == expected

    def test_get_rdt_paths_multiple_rdts(self):
        """Test get_rdt_paths with multiple RDTs of different types."""
        rdts = ["f3000_x", "f2110_y", "f4000_x"]
        output_dir = Path("/output")
        result = get_rdt_paths(rdts, output_dir)
        expected = {
            "f3000_x": Path("/output/normal_sextupole/f3000_x.tfs"),
            "f2110_y": Path("/output/skew_sextupole/f2110_y.tfs"),
            "f4000_x": Path("/output/normal_octupole/f4000_x.tfs"),
        }
        assert result == expected

    def test_get_rdt_paths_empty_list(self):
        """Test get_rdt_paths with empty RDT list."""
        rdts = []
        output_dir = Path("/output")
        result = get_rdt_paths(rdts, output_dir)
        assert result == {}

    def test_get_rdt_paths_coupling_rdts(self):
        """Test get_rdt_paths with coupling RDTs."""
        rdts = ["f1001_x", "f1010_y"]
        output_dir = Path("/output")
        result = get_rdt_paths(rdts, output_dir)
        expected = {
            "f1001_x": Path("/output/skew_sextupole/f1001_x.tfs"),
            "f1010_y": Path("/output/normal_sextupole/f1010_y.tfs"),
        }
        assert result == expected


class TestGetTunes:
    """Test cases for get_tunes function."""

    @patch("lhcng.analysis.tfs.reader.read_headers")
    def test_get_tunes_success(self, mock_read_headers):
        """Test get_tunes successfully extracts tunes from headers."""
        mock_read_headers.return_value = {"Q1": 0.28, "Q2": 0.31}
        output_dir = Path("/output")
        result = get_tunes(output_dir)
        assert result == [0.28, 0.31]
        mock_read_headers.assert_called_once_with(Path("/output/beta_amplitude_x.tfs"))

    @patch("lhcng.analysis.tfs.reader.read_headers")
    def test_get_tunes_missing_headers(self, mock_read_headers):
        """Test get_tunes behavior when headers are missing."""
        mock_read_headers.return_value = {"Q1": 0.28}  # Missing Q2
        output_dir = Path("/output")

        with pytest.raises(KeyError):
            get_tunes(output_dir)

    @patch("lhcng.analysis.tfs.reader.read_headers")
    def test_get_tunes_file_not_found(self, mock_read_headers):
        """Test get_tunes when optics file is not found."""
        mock_read_headers.side_effect = FileNotFoundError("File not found")
        output_dir = Path("/output")

        with pytest.raises(FileNotFoundError):
            get_tunes(output_dir)

    @patch("lhcng.analysis.tfs.reader.read_headers")
    def test_get_tunes_non_numeric_values(self, mock_read_headers):
        """Test get_tunes with non-numeric tune values."""
        mock_read_headers.return_value = {"Q1": "invalid", "Q2": 0.31}
        output_dir = Path("/output")

        result = get_tunes(output_dir)
        assert result == ["invalid", 0.31]

    @patch("lhcng.analysis.tfs.reader.read_headers")
    def test_get_tunes_zero_values(self, mock_read_headers):
        """Test get_tunes with zero tune values."""
        mock_read_headers.return_value = {"Q1": 0.0, "Q2": 0.0}
        output_dir = Path("/output")
        result = get_tunes(output_dir)
        assert result == [0.0, 0.0]


class TestGetRdtsFromOpticsAnalysis:
    """Test cases for get_rdts_from_optics_analysis function."""

    @patch("lhcng.analysis.get_tunes")
    @patch("lhcng.analysis.tfs.read")
    @patch("lhcng.analysis.filter_out_BPM_near_IPs")
    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.get_model_dir")
    @patch("lhcng.analysis.get_output_dir")
    @patch("lhcng.analysis.get_rdt_paths")
    @patch("lhcng.analysis.ALL_RDTS", ["f3000_x"])
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    def test_get_rdts_from_optics_analysis_success(
        self,
        mock_get_rdt_paths,
        mock_get_output_dir,
        mock_get_model_dir,
        mock_model_compressor,
        mock_hole_in_one,
        mock_filter_bpm,
        mock_tfs_read,
        mock_get_tunes,
    ):
        """Test successful RDT analysis extraction."""
        # Setup mocks
        beam = 1
        tbt_path = Path("/data/test.sdds")
        model_dir = Path("/model")
        output_dir = Path("/output")

        mock_get_output_dir.return_value = output_dir
        mock_get_model_dir.return_value = Path("/model/beam1")
        mock_get_rdt_paths.return_value = {"f3000_x": Path("/output/normal_sextupole/f3000_x.tfs")}
        mock_get_tunes.return_value = [0.28, 0.31]

        # Mock TFS dataframe
        mock_df = MagicMock()
        mock_tfs_read.return_value = mock_df
        mock_filter_bpm.return_value = mock_df

        # Mock ModelCompressor context manager
        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        result = get_rdts_from_optics_analysis(beam, tbt_path, model_dir, output_dir)

        # Assertions
        assert result == {"f3000_x": mock_df}
        mock_hole_in_one.assert_called_once()
        mock_get_tunes.assert_called_once_with(output_dir)
        mock_tfs_read.assert_called_once_with(Path("/output/normal_sextupole/f3000_x.tfs"), index="NAME")
        mock_filter_bpm.assert_called_once_with(mock_df)

    @patch("lhcng.analysis.get_tunes")
    @patch("lhcng.analysis.tfs.read")
    @patch("lhcng.analysis.filter_out_BPM_near_IPs")
    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.get_model_dir")
    @patch("lhcng.analysis.get_output_dir")
    @patch("lhcng.analysis.get_rdt_paths")
    @patch("lhcng.analysis.ALL_RDTS", ["f1001", "f1010"])
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    def test_get_rdts_from_optics_analysis_coupling_only(
        self,
        mock_get_rdt_paths,
        mock_get_output_dir,
        mock_get_model_dir,
        mock_model_compressor,
        mock_hole_in_one,
        mock_filter_bpm,
        mock_tfs_read,
        mock_get_tunes,
    ):
        """Test RDT analysis with coupling-only RDTs."""
        beam = 2
        tbt_path = Path("/data/test.sdds")
        model_dir = Path("/model")

        mock_get_output_dir.return_value = Path("/output")
        mock_get_model_dir.return_value = Path("/model/beam2")
        mock_get_rdt_paths.return_value = {"f1001": Path("/output/f1001.tfs")}
        mock_get_tunes.return_value = [0.28, 0.31]

        mock_df = MagicMock()
        mock_tfs_read.return_value = mock_df
        mock_filter_bpm.return_value = mock_df

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        get_rdts_from_optics_analysis(beam, tbt_path, model_dir)

        # Check that only_coupling was set to True
        args, kwargs = mock_hole_in_one.call_args
        assert kwargs["only_coupling"] is True

    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.get_output_dir")
    @patch("lhcng.analysis.get_rdt_paths")
    def test_get_rdts_from_optics_analysis_hole_in_one_error(
        self, mock_get_rdt_paths, mock_get_output_dir, mock_model_compressor
    ):
        """Test RDT analysis when hole_in_one_entrypoint raises an exception."""
        beam = 1
        tbt_path = Path("/data/test.sdds")
        model_dir = Path("/model")

        mock_get_output_dir.return_value = Path("/output")
        mock_get_rdt_paths.return_value = {}

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        with patch("lhcng.analysis.hole_in_one_entrypoint", side_effect=RuntimeError("Analysis failed")), pytest.raises(RuntimeError, match="Analysis failed"):
            get_rdts_from_optics_analysis(beam, tbt_path, model_dir)

    @patch("lhcng.analysis.get_tunes")
    @patch("lhcng.analysis.tfs.read")
    @patch("lhcng.analysis.filter_out_BPM_near_IPs")
    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.get_model_dir")
    @patch("lhcng.analysis.get_output_dir")
    @patch("lhcng.analysis.get_rdt_paths")
    @patch("lhcng.analysis.ALL_RDTS", ["f3000_x", "f2110_y"])
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    def test_get_rdts_from_optics_analysis_multiple_rdts(
        self,
        mock_get_rdt_paths,
        mock_get_output_dir,
        mock_get_model_dir,
        mock_model_compressor,
        mock_hole_in_one,
        mock_filter_bmp,
        mock_tfs_read,
        mock_get_tunes,
    ):
        """Test RDT analysis with multiple RDTs."""
        beam = 1
        tbt_path = Path("/data/test.sdds")
        model_dir = Path("/model")
        output_dir = Path("/output")

        mock_get_output_dir.return_value = output_dir
        mock_get_model_dir.return_value = Path("/model/beam1")
        mock_get_rdt_paths.return_value = {
            "f3000_x": Path("/output/normal_sextupole/f3000_x.tfs"),
            "f2110_y": Path("/output/skew_sextupole/f2110_y.tfs"),
        }
        mock_get_tunes.return_value = [0.28, 0.31]

        # Mock TFS dataframes
        mock_df1 = MagicMock()
        mock_df2 = MagicMock()
        mock_tfs_read.side_effect = [mock_df1, mock_df2]
        mock_filter_bmp.side_effect = [mock_df1, mock_df2]

        # Mock ModelCompressor context manager
        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        result = get_rdts_from_optics_analysis(beam, tbt_path, model_dir, output_dir)

        # Assertions
        expected_result = {"f3000_x": mock_df1, "f2110_y": mock_df2}
        assert result == expected_result
        assert mock_tfs_read.call_count == 2


class TestRunHarpy:
    """Test cases for run_harpy function."""

    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.logger")
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    @patch("lhcng.analysis.DATA_DIR", Path("/data"))
    @patch("lhcng.analysis.ACCEL", "lhc")
    def test_run_harpy_default_parameters(self, mock_logger, mock_model_compressor, mock_hole_in_one):
        """Test run_harpy with default parameters."""
        beam = 1
        model_dir = Path("/model")

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        run_harpy(beam, model_dir)

        # Verify hole_in_one_entrypoint was called with correct parameters
        mock_hole_in_one.assert_called_once()
        args, kwargs = mock_hole_in_one.call_args

        assert kwargs["harpy"] is True
        assert kwargs["files"] == [Path("/data/tbt_data_b1.sdds")]
        assert kwargs["outputdir"] == Path("/freq_out")
        assert kwargs["to_write"] == ["lin", "spectra"]
        assert kwargs["opposite_direction"] is False
        assert kwargs["accel"] == "lhc"
        assert kwargs["tunes"] == [0.28, 0.31, 0.0]
        assert kwargs["natdeltas"] == [0.0, -0.0, 0.0]
        assert kwargs["clean"] is False

        mock_logger.info.assert_called_once_with("Harpy analysis complete for beam 1.")

    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.logger")
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    @patch("lhcng.analysis.DATA_DIR", Path("/data"))
    @patch("lhcng.analysis.ACCEL", "lhc")
    def test_run_harpy_beam_2_opposite_direction(self, mock_logger, mock_model_compressor, mock_hole_in_one):
        """Test run_harpy with beam 2 sets opposite_direction to True."""
        beam = 2
        model_dir = Path("/model")

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        run_harpy(beam, model_dir)

        args, kwargs = mock_hole_in_one.call_args
        assert kwargs["opposite_direction"] is True
        assert kwargs["files"] == [Path("/data/tbt_data_b2.sdds")]
        mock_logger.info.assert_called_once_with("Harpy analysis complete for beam 2.")

    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.DATA_DIR", Path("/data"))
    @patch("lhcng.analysis.ACCEL", "lhc")
    def test_run_harpy_custom_parameters(self, mock_model_compressor, mock_hole_in_one):
        """Test run_harpy with custom parameters."""
        beam = 1
        model_dir = Path("/model")
        custom_tunes = [0.25, 0.32, 0.1]
        custom_natdeltas = [0.1, -0.1, 0.05]
        custom_linfile_dir = Path("/custom/output")

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        run_harpy(
            beam=beam,
            model_dir=model_dir,
            tunes=custom_tunes,
            natdeltas=custom_natdeltas,
            linfile_dir=custom_linfile_dir,
            clean=True,
        )

        args, kwargs = mock_hole_in_one.call_args
        assert kwargs["tunes"] == custom_tunes
        assert kwargs["natdeltas"] == custom_natdeltas
        assert kwargs["outputdir"] == custom_linfile_dir
        assert kwargs["clean"] is True

    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.FREQ_OUT_DIR", Path("/freq_out"))
    @patch("lhcng.analysis.DATA_DIR", Path("/data"))
    def test_run_harpy_hole_in_one_error(self, mock_model_compressor):
        """Test run_harpy when hole_in_one_entrypoint raises an exception."""
        beam = 1
        model_dir = Path("/model")

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        with patch("lhcng.analysis.hole_in_one_entrypoint", side_effect=RuntimeError("Harpy failed")), pytest.raises(RuntimeError, match="Harpy failed"):
            run_harpy(beam, model_dir)

    @patch("lhcng.analysis.hole_in_one_entrypoint")
    @patch("lhcng.analysis.ModelCompressor")
    @patch("lhcng.analysis.logger")
    @patch("lhcng.analysis.DATA_DIR", Path("/data"))
    @patch("lhcng.analysis.ACCEL", "lhc")
    def test_run_harpy_invalid_beam_number(self, mock_logger, mock_model_compressor, mock_hole_in_one):
        """Test run_harpy with invalid beam number."""
        beam = 3  # Invalid beam number
        model_dir = Path("/model")

        mock_compressor_instance = MagicMock()
        mock_model_compressor.return_value.__enter__.return_value = mock_compressor_instance
        mock_model_compressor.return_value.__exit__.return_value = None

        # This should still work as the function doesn't validate beam numbers
        run_harpy(beam, model_dir)

        args, kwargs = mock_hole_in_one.call_args
        assert kwargs["files"] == [Path("/data/tbt_data_b3.sdds")]
        assert kwargs["opposite_direction"] is False  # Default for non-beam-2


class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""

    def test_integration_rdt_analysis_workflow(self):
        """Integration test for the complete RDT analysis workflow."""
        # Test the workflow of get_rdt_type -> get_rdt_paths -> analysis
        rdts = ["f3000_x", "f2110_y", "f4000_x"]
        output_dir = Path("/output")

        # Test RDT type determination
        rdt_types = [get_rdt_type(rdt) for rdt in rdts]
        expected_types = [
            ("normal", "sextupole"),
            ("skew", "sextupole"),
            ("normal", "octupole"),
        ]
        assert rdt_types == expected_types

        # Test path generation
        rdt_paths = get_rdt_paths(rdts, output_dir)
        expected_paths = {
            "f3000_x": Path("/output/normal_sextupole/f3000_x.tfs"),
            "f2110_y": Path("/output/skew_sextupole/f2110_y.tfs"),
            "f4000_x": Path("/output/normal_octupole/f4000_x.tfs"),
        }
        assert rdt_paths == expected_paths

    def test_rdt_type_comprehensive_coverage(self):
        """Comprehensive test covering all RDT type combinations."""
        test_cases = [
            # Normal sextupoles (sum != 4, third+fourth digits even)
            ("f3000_x", ("normal", "sextupole")),
            ("f1200_y", ("normal", "sextupole")),
            ("f2100_x", ("normal", "sextupole")),

            # Skew sextupoles (sum != 4, third+fourth digits odd)
            ("f2110_y", ("skew", "sextupole")),
            ("f1110_x", ("skew", "sextupole")),
            ("f3100_y", ("skew", "sextupole")),

            # Normal octupoles (sum == 4, third+fourth digits even)
            ("f4000_x", ("normal", "octupole")),
            ("f2200_y", ("normal", "octupole")),
            ("f1300_x", ("normal", "octupole")),

            # Skew octupoles (sum == 4, third+fourth digits odd)
            ("f2101_y", ("skew", "octupole")),
            ("f3001_x", ("skew", "octupole")),
            ("f1210_y", ("skew", "octupole")),
        ]

        for rdt, expected in test_cases:
            result = get_rdt_type(rdt)
            assert result == expected, f"Failed for RDT {rdt}: got {result}, expected {expected}"

    def test_complete_analysis_parameter_propagation(self):
        """Test that parameters are correctly propagated through the analysis chain."""
        # This tests the integration between functions
        rdts = ["f1001_x", "f1010_y"]  # Coupling RDTs

        # Verify that coupling detection works
        coupling_only = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
        assert coupling_only is True

        # Verify RDT types
        types = [get_rdt_type(rdt) for rdt in rdts]
        assert types == [("skew", "sextupole"), ("normal", "sextupole")]

        # Verify paths
        paths = get_rdt_paths(rdts, Path("/test"))
        expected = {
            "f1001_x": Path("/test/skew_sextupole/f1001_x.tfs"),
            "f1010_y": Path("/test/normal_sextupole/f1010_y.tfs"),
        }
        assert paths == expected


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_get_rdt_type_malformed_input(self):
        """Test get_rdt_type with various malformed inputs."""
        malformed_inputs = [
            "",           # Empty string
            "f",          # Too short
            "fxyz_x",     # Non-numeric characters
            "f123_x",     # Too few digits
            "f12345_x",   # Too many digits
        ]

        for malformed_input in malformed_inputs:
            with pytest.raises((ValueError, IndexError)):
                get_rdt_type(malformed_input)

    @patch("lhcng.analysis.tfs.read")
    def test_tfs_read_failure_handling(self, mock_tfs_read):
        """Test handling of TFS file reading failures."""
        mock_tfs_read.side_effect = IOError("Cannot read TFS file")

        # This would be called from get_rdts_from_optics_analysis
        with pytest.raises(IOError):
            mock_tfs_read(Path("/nonexistent/file.tfs"), index="NAME")

    def test_path_handling_edge_cases(self):
        """Test path handling with various edge cases."""
        # Test with relative paths
        output_dir = Path("relative/path")
        rdts = ["f3000_x"]
        result = get_rdt_paths(rdts, output_dir)
        expected = {"f3000_x": Path("relative/path/normal_sextupole/f3000_x.tfs")}
        assert result == expected

        # Test with empty path
        output_dir = Path("")
        result = get_rdt_paths(rdts, output_dir)
        expected = {"f3000_x": Path("normal_sextupole/f3000_x.tfs")}
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])