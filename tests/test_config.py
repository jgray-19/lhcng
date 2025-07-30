"""
Comprehensive unit tests for the config.py module.

Testing Framework: pytest
This module tests configuration settings, directory creation, environment variable handling,
and RDT definitions for the lhcng package.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Import the config module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class TestAcceleratorConfiguration:
    """Test accelerator name configuration and environment variable handling."""
    
    def test_default_accel_is_lhc(self):
        """Test that the default accelerator is 'lhc' when ACCEL env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ACCEL if it exists
            if 'ACCEL' in os.environ:
                del os.environ['ACCEL']
            # Reload the module to test default behavior
            import importlib
            importlib.reload(config)
            assert config.ACCEL == "lhc"
    
    def test_accel_from_environment_variable(self):
        """Test that ACCEL is set from environment variable when provided."""
        test_accel = "ps"
        with patch.dict(os.environ, {'ACCEL': test_accel}):
            import importlib
            importlib.reload(config)
            assert test_accel == config.ACCEL
    
    def test_accel_environment_variable_override(self):
        """Test that environment variable overrides the default value."""
        test_accels = ["ps", "sps", "leir", "elena", "ad"]
        for test_accel in test_accels:
            with patch.dict(os.environ, {'ACCEL': test_accel}):
                import importlib
                importlib.reload(config)
                assert test_accel == config.ACCEL
    
    def test_accel_empty_string_environment_variable(self):
        """Test behavior when ACCEL environment variable is empty string."""
        with patch.dict(os.environ, {'ACCEL': ''}):
            import importlib
            importlib.reload(config)
            assert config.ACCEL == ""
    
    def test_accel_whitespace_environment_variable(self):
        """Test behavior when ACCEL environment variable contains whitespace."""
        with patch.dict(os.environ, {'ACCEL': '  lhc  '}):
            import importlib
            importlib.reload(config)
            assert config.ACCEL == "  lhc  "


class TestDirectoryConfiguration:
    """Test directory path configuration and creation."""
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create a temporary working directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)
    
    def test_current_dir_is_resolved_path(self, temp_working_dir):
        """Test that CURRENT_DIR is the resolved current working directory."""
        import importlib
        importlib.reload(config)
        assert temp_working_dir.resolve() == config.CURRENT_DIR
        assert config.CURRENT_DIR.is_absolute()
    
    def test_analysis_dir_path_construction(self, temp_working_dir):
        """Test that ANALYSIS_DIR is correctly constructed relative to CURRENT_DIR."""
        import importlib
        importlib.reload(config)
        expected_path = temp_working_dir / "analysis"
        assert expected_path == config.ANALYSIS_DIR
    
    def test_freq_out_dir_path_construction(self, temp_working_dir):
        """Test that FREQ_OUT_DIR is correctly nested under ANALYSIS_DIR."""
        import importlib
        importlib.reload(config)
        expected_path = temp_working_dir / "analysis" / "lin_files"
        assert expected_path == config.FREQ_OUT_DIR
    
    def test_data_dir_path_construction(self, temp_working_dir):
        """Test that DATA_DIR is correctly constructed relative to CURRENT_DIR."""
        import importlib
        importlib.reload(config)
        expected_path = temp_working_dir / "data"
        assert expected_path == config.DATA_DIR
    
    def test_plot_dir_path_construction(self, temp_working_dir):
        """Test that PLOT_DIR is correctly constructed relative to CURRENT_DIR."""
        import importlib
        importlib.reload(config)
        expected_path = temp_working_dir / "plots"
        assert expected_path == config.PLOT_DIR
    
    def test_acc_models_dir_with_default_accel(self, temp_working_dir):
        """Test that ACC_MODELS path includes the accelerator name with default ACCEL."""
        with patch.dict(os.environ, {}, clear=True):
            if 'ACCEL' in os.environ:
                del os.environ['ACCEL']
            import importlib
            importlib.reload(config)
            expected_path = temp_working_dir / "acc-models-lhc"
            assert expected_path == config.ACC_MODELS
    
    def test_acc_models_dir_with_custom_accel(self, temp_working_dir):
        """Test that ACC_MODELS path includes custom accelerator name."""
        custom_accel = "ps"
        with patch.dict(os.environ, {'ACCEL': custom_accel}):
            import importlib
            importlib.reload(config)
            expected_path = temp_working_dir / f"acc-models-{custom_accel}"
            assert expected_path == config.ACC_MODELS
    
    def test_directories_are_created_on_import(self, temp_working_dir):
        """Test that required directories are created when the module is imported."""
        # Ensure directories don't exist initially
        analysis_dir = temp_working_dir / "analysis"
        freq_out_dir = temp_working_dir / "analysis" / "lin_files"
        data_dir = temp_working_dir / "data"
        plot_dir = temp_working_dir / "plots"
        
        for directory in [analysis_dir, freq_out_dir, data_dir, plot_dir]:
            if directory.exists():
                shutil.rmtree(directory)
        
        # Import the module (which should create directories)
        import importlib
        importlib.reload(config)
        
        # Verify directories were created
        assert analysis_dir.exists()
        assert analysis_dir.is_dir()
        assert freq_out_dir.exists()
        assert freq_out_dir.is_dir()
        assert data_dir.exists()
        assert data_dir.is_dir()
        assert plot_dir.exists()
        assert plot_dir.is_dir()
    
    def test_directories_creation_with_existing_dirs(self, temp_working_dir):
        """Test that directory creation works when directories already exist."""
        # Pre-create directories
        analysis_dir = temp_working_dir / "analysis"
        data_dir = temp_working_dir / "data"
        analysis_dir.mkdir()
        data_dir.mkdir()
        
        # Import should not fail
        import importlib
        importlib.reload(config)
        
        # Directories should still exist
        assert analysis_dir.exists()
        assert data_dir.exists()
    
    def test_nested_directory_creation(self, temp_working_dir):
        """Test that nested directories (like FREQ_OUT_DIR) are created properly."""
        import importlib
        importlib.reload(config)
        
        freq_out_dir = temp_working_dir / "analysis" / "lin_files"
        assert freq_out_dir.exists()
        assert freq_out_dir.is_dir()
        assert freq_out_dir.parent.exists()  # parent (analysis) should also exist


class TestRDTDefinitions:
    """Test RDT (Resonance Driving Terms) definitions and constants."""
    
    def test_normal_sextupole_rdts_structure(self):
        """Test that NORMAL_SEXTUPOLE_RDTS is properly defined."""
        assert isinstance(config.NORMAL_SEXTUPOLE_RDTS, tuple)
        assert len(config.NORMAL_SEXTUPOLE_RDTS) == 8
        
        expected_rdts = [
            "f1200_x", "f3000_x", "f1002_x", "f1020_x",
            "f0111_y", "f0120_y", "f1011_y", "f1020_y"
        ]
        assert all(rdt in config.NORMAL_SEXTUPOLE_RDTS for rdt in expected_rdts)
    
    def test_skew_sextupole_rdts_structure(self):
        """Test that SKEW_SEXTUPOLE_RDTS is properly defined."""
        assert isinstance(config.SKEW_SEXTUPOLE_RDTS, tuple)
        assert len(config.SKEW_SEXTUPOLE_RDTS) == 8
        
        expected_rdts = [
            "f0012_y", "f0030_y", "f1101_x", "f1110_x",
            "f2001_x", "f2010_x", "f0210_y", "f2010_y"
        ]
        assert all(rdt in config.SKEW_SEXTUPOLE_RDTS for rdt in expected_rdts)
    
    def test_normal_octupole_rdts_structure(self):
        """Test that NORMAL_OCTUPOLE_RDTS is properly defined."""
        assert isinstance(config.NORMAL_OCTUPOLE_RDTS, tuple)
        assert len(config.NORMAL_OCTUPOLE_RDTS) == 12
        
        expected_rdts = [
            "f1300_x", "f4000_x", "f0013_y", "f0040_y",
            "f1102_x", "f1120_x", "f2002_x", "f2020_x",
            "f0211_y", "f0220_y", "f2011_y", "f2020_y"
        ]
        assert all(rdt in config.NORMAL_OCTUPOLE_RDTS for rdt in expected_rdts)
    
    def test_skew_octupole_rdts_structure(self):
        """Test that SKEW_OCTUPOLE_RDTS is properly defined."""
        assert isinstance(config.SKEW_OCTUPOLE_RDTS, tuple)
        assert len(config.SKEW_OCTUPOLE_RDTS) == 13
        
        expected_rdts = [
            "f0112_y", "f0130_y", "f0310_y", "f1003_x",
            "f1012_y", "f1030_x", "f1030_y", "f1201_x",
            "f1210_x", "f3001_x", "f3010_x", "f3010_y"
        ]
        assert all(rdt in config.SKEW_OCTUPOLE_RDTS for rdt in expected_rdts)
    
    def test_coupling_rdts_structure(self):
        """Test that COUPLING_RDTS is properly defined."""
        assert isinstance(config.COUPLING_RDTS, list)
        assert len(config.COUPLING_RDTS) == 2
        assert "f1001" in config.COUPLING_RDTS
        assert "f1010" in config.COUPLING_RDTS
    
    def test_all_rdts_combination(self):
        """Test that ALL_RDTS properly combines all RDT tuples."""
        assert isinstance(config.ALL_RDTS, tuple)
        
        expected_length = (
            len(config.NORMAL_SEXTUPOLE_RDTS) +
            len(config.SKEW_SEXTUPOLE_RDTS) +
            len(config.NORMAL_OCTUPOLE_RDTS) +
            len(config.SKEW_OCTUPOLE_RDTS)
        )
        assert len(config.ALL_RDTS) == expected_length
        
        # Verify all individual RDT collections are included
        for rdt in config.NORMAL_SEXTUPOLE_RDTS:
            assert rdt in config.ALL_RDTS
        for rdt in config.SKEW_SEXTUPOLE_RDTS:
            assert rdt in config.ALL_RDTS
        for rdt in config.NORMAL_OCTUPOLE_RDTS:
            assert rdt in config.ALL_RDTS
        for rdt in config.SKEW_OCTUPOLE_RDTS:
            assert rdt in config.ALL_RDTS
    
    def test_rdt_naming_conventions(self):
        """Test that RDT names follow expected conventions."""
        all_rdts = list(config.ALL_RDTS)
        
        # Test that all RDTs start with 'f'
        assert all(rdt.startswith('f') for rdt in all_rdts)
        
        # Test that all RDTs contain only valid characters
        import re
        valid_pattern = re.compile(r'^f[0-9]{4}_[xy]$')
        assert all(valid_pattern.match(rdt) for rdt in all_rdts)
    
    def test_rdt_uniqueness(self):
        """Test that there are no duplicate RDTs in ALL_RDTS."""
        all_rdts = list(config.ALL_RDTS)
        assert len(all_rdts) == len(set(all_rdts))
    
    def test_rdt_immutability(self):
        """Test that RDT tuples are immutable."""
        with pytest.raises(TypeError):
            config.NORMAL_SEXTUPOLE_RDTS[0] = "modified"
        
        with pytest.raises(TypeError):
            config.SKEW_SEXTUPOLE_RDTS.append("new_rdt")


class TestModuleConstants:
    """Test module-level constants and their properties."""
    
    def test_all_path_objects_are_pathlib_paths(self):
        """Test that all directory constants are pathlib.Path objects."""
        path_constants = [
            config.CURRENT_DIR,
            config.ANALYSIS_DIR,
            config.FREQ_OUT_DIR,
            config.DATA_DIR,
            config.ACC_MODELS,
            config.PLOT_DIR
        ]
        
        for path_const in path_constants:
            assert isinstance(path_const, Path)
    
    def test_path_relationships(self):
        """Test that directory paths have correct parent-child relationships."""
        # FREQ_OUT_DIR should be child of ANALYSIS_DIR
        assert config.FREQ_OUT_DIR.parent == config.ANALYSIS_DIR
        
        # All main directories should be children of CURRENT_DIR
        main_dirs = [config.ANALYSIS_DIR, config.DATA_DIR, config.PLOT_DIR, config.ACC_MODELS]
        for directory in main_dirs:
            assert directory.parent == config.CURRENT_DIR
    
    def test_string_constants_are_strings(self):
        """Test that string constants have correct types."""
        assert isinstance(config.ACCEL, str)
    
    def test_module_docstring_exists(self):
        """Test that the module has proper documentation."""
        assert config.__doc__ is not None
        assert len(config.__doc__.strip()) > 0
        assert "lhcng" in config.__doc__
        assert "configuration" in config.__doc__.lower()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_config_with_special_characters_in_accel(self):
        """Test behavior with special characters in ACCEL environment variable."""
        special_accels = ["lhc-test", "lhc_test", "lhc.test", "lhc@test"]
        
        for special_accel in special_accels:
            with patch.dict(os.environ, {'ACCEL': special_accel}):
                import importlib
                importlib.reload(config)
                assert special_accel == config.ACCEL
                # ACC_MODELS should include the special character
                assert str(config.ACC_MODELS).endswith(f"acc-models-{special_accel}")
    
    def test_config_with_unicode_in_accel(self):
        """Test behavior with Unicode characters in ACCEL environment variable."""
        unicode_accel = "lhc_été"
        with patch.dict(os.environ, {'ACCEL': unicode_accel}):
            import importlib
            importlib.reload(config)
            assert unicode_accel == config.ACCEL
    
    @patch('config.Path.mkdir')
    def test_directory_creation_failure_handling(self, mock_mkdir):
        """Test behavior when directory creation fails."""
        # Mock mkdir to raise an exception
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        # The module should still import, but directories won't be created
        with pytest.raises(PermissionError):
            import importlib
            importlib.reload(config)
    
    def test_config_reload_consistency(self):
        """Test that reloading the config module maintains consistency."""
        original_accel = config.ACCEL
        original_current_dir = config.CURRENT_DIR
        
        import importlib
        importlib.reload(config)
        
        # Core values should remain consistent
        assert original_accel == config.ACCEL
        assert original_current_dir == config.CURRENT_DIR


class TestIntegrationScenarios:
    """Test integration scenarios and realistic usage patterns."""
    
    def test_typical_lhc_configuration(self):
        """Test typical LHC configuration scenario."""
        with patch.dict(os.environ, {'ACCEL': 'lhc'}):
            import importlib
            importlib.reload(config)
            
            assert config.ACCEL == "lhc"
            assert "acc-models-lhc" in str(config.ACC_MODELS)
            assert config.ANALYSIS_DIR.name == "analysis"
            assert config.FREQ_OUT_DIR.name == "lin_files"
    
    def test_alternative_accelerator_configuration(self):
        """Test configuration for alternative accelerators."""
        alt_accelerators = ["ps", "sps", "leir", "elena"]
        
        for accel in alt_accelerators:
            with patch.dict(os.environ, {'ACCEL': accel}):
                import importlib
                importlib.reload(config)
                
                assert accel == config.ACCEL
                assert f"acc-models-{accel}" in str(config.ACC_MODELS)
    
    def test_all_rdt_collections_are_accessible(self):
        """Test that all RDT collections can be accessed and used."""
        rdt_collections = [
            config.NORMAL_SEXTUPOLE_RDTS,
            config.SKEW_SEXTUPOLE_RDTS,
            config.NORMAL_OCTUPOLE_RDTS,
            config.SKEW_OCTUPOLE_RDTS,
            config.COUPLING_RDTS,
            config.ALL_RDTS
        ]
        
        for collection in rdt_collections:
            assert len(collection) > 0
            # Test that we can iterate over each collection
            for rdt in collection:
                assert isinstance(rdt, str)
                assert len(rdt) > 0


if __name__ == "__main__":
    pytest.main([__file__])