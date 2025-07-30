"""
Unit tests for the tracking module.

This test suite covers the tracking simulation functions including:
- File path generation utilities
- Tracking simulation execution
- Orbit correction functionality
- Edge cases and error conditions

Testing Framework: pytest (based on project structure)
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from src.tracking import (
    get_file_suffix,
    get_tfs_path,
    get_tbt_path,
    correct_orbit,
    run_tracking,
)


class TestGetFileSuffix:
    """Test suite for get_file_suffix function."""
    
    def test_get_file_suffix_beam1_default_params(self):
        """Test file suffix generation for beam 1 with default parameters."""
        result = get_file_suffix(beam=1, nturns=1000)
        expected = "b1_c0_t0.28_0.31_t1000_k0.001"
        assert result == expected
    
    def test_get_file_suffix_beam2_default_params(self):
        """Test file suffix generation for beam 2 with default parameters."""
        result = get_file_suffix(beam=2, nturns=500)
        expected = "b2_c0_t0.28_0.31_t500_k0.001"
        assert result == expected
    
    def test_get_file_suffix_with_coupling_boolean(self):
        """Test file suffix generation with boolean coupling."""
        result = get_file_suffix(beam=1, nturns=100, coupling_knob=True)
        expected = "b1_c1_t0.28_0.31_t100_k0.001"
        assert result == expected
    
    def test_get_file_suffix_with_coupling_float(self):
        """Test file suffix generation with float coupling value."""
        result = get_file_suffix(beam=2, nturns=200, coupling_knob=0.5)
        expected = "b2_c0.5_t0.28_0.31_t200_k0.001"
        assert result == expected
    
    def test_get_file_suffix_custom_tunes(self):
        """Test file suffix generation with custom tunes."""
        result = get_file_suffix(beam=1, nturns=300, tunes=[0.30, 0.32])
        expected = "b1_c0_t0.3_0.32_t300_k0.001"
        assert result == expected
    
    def test_get_file_suffix_custom_kick_amp(self):
        """Test file suffix generation with custom kick amplitude."""
        result = get_file_suffix(beam=2, nturns=400, kick_amp=2e-3)
        expected = "b2_c0_t0.28_0.31_t400_k0.002"
        assert result == expected
    
    def test_get_file_suffix_all_custom_params(self):
        """Test file suffix generation with all custom parameters."""
        result = get_file_suffix(
            beam=1, 
            nturns=750, 
            coupling_knob=0.25, 
            tunes=[0.29, 0.33], 
            kick_amp=1.5e-3
        )
        expected = "b1_c0.25_t0.29_0.33_t750_k0.0015"
        assert result == expected
    
    def test_get_file_suffix_invalid_beam_raises_assertion_error(self):
        """Test that invalid beam number raises AssertionError."""
        with pytest.raises(AssertionError, match="Beam must be 1 or 2"):
            get_file_suffix(beam=3, nturns=100)
        
        with pytest.raises(AssertionError, match="Beam must be 1 or 2"):
            get_file_suffix(beam=0, nturns=100)
    
    def test_get_file_suffix_zero_turns(self):
        """Test file suffix generation with zero turns."""
        result = get_file_suffix(beam=1, nturns=0)
        expected = "b1_c0_t0.28_0.31_t0_k0.001"
        assert result == expected
    
    def test_get_file_suffix_negative_turns(self):
        """Test file suffix generation with negative turns."""
        result = get_file_suffix(beam=1, nturns=-100)
        expected = "b1_c0_t0.28_0.31_t-100_k0.001"
        assert result == expected
    
    def test_get_file_suffix_large_numbers(self):
        """Test file suffix generation with large numbers."""
        result = get_file_suffix(beam=2, nturns=1000000, kick_amp=1e-6)
        expected = "b2_c0_t0.28_0.31_t1000000_k1e-06"
        assert result == expected


class TestGetTfsPath:
    """Test suite for get_tfs_path function."""
    
    @patch('src.tracking.DATA_DIR', Path('/test/data'))
    def test_get_tfs_path_default_params(self):
        """Test TFS path generation with default parameters."""
        result = get_tfs_path(beam=1, nturns=1000)
        expected = Path('/test/data/b1_c0_t0.28_0.31_t1000_k0.001.tfs.bz2')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/custom/path'))
    def test_get_tfs_path_custom_data_dir(self):
        """Test TFS path generation with custom data directory."""
        result = get_tfs_path(beam=2, nturns=500)
        expected = Path('/custom/path/b2_c0_t0.28_0.31_t500_k0.001.tfs.bz2')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tfs_path_with_coupling(self):
        """Test TFS path generation with coupling enabled."""
        result = get_tfs_path(beam=1, nturns=200, coupling_knob=True)
        expected = Path('/data/b1_c1_t0.28_0.31_t200_k0.001.tfs.bz2')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tfs_path_all_custom_params(self):
        """Test TFS path generation with all custom parameters."""
        result = get_tfs_path(
            beam=2, 
            nturns=300, 
            coupling_knob=0.7, 
            tunes=[0.25, 0.35], 
            kick_amp=5e-4
        )
        expected = Path('/data/b2_c0.7_t0.25_0.35_t300_k0.0005.tfs.bz2')
        assert result == expected


class TestGetTbtPath:
    """Test suite for get_tbt_path function."""
    
    @patch('src.tracking.DATA_DIR', Path('/test/data'))
    def test_get_tbt_path_default_params(self):
        """Test TBT path generation with default parameters."""
        result = get_tbt_path(beam=1, nturns=1000)
        expected = Path('/test/data/tbt_b1_c0_t0.28_0.31_t1000_k0.001_0.sdds')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tbt_path_custom_index_int(self):
        """Test TBT path generation with custom integer index."""
        result = get_tbt_path(beam=2, nturns=500, index=5)
        expected = Path('/data/tbt_b2_c0_t0.28_0.31_t500_k0.001_5.sdds')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tbt_path_custom_index_string(self):
        """Test TBT path generation with custom string index."""
        result = get_tbt_path(beam=1, nturns=300, index="abc")
        expected = Path('/data/tbt_b1_c0_t0.28_0.31_t300_k0.001_abc.sdds')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tbt_path_with_coupling_and_custom_params(self):
        """Test TBT path generation with coupling and custom parameters."""
        result = get_tbt_path(
            beam=2, 
            nturns=750, 
            coupling_knob=0.3, 
            tunes=[0.27, 0.32], 
            kick_amp=8e-4, 
            index=10
        )
        expected = Path('/data/tbt_b2_c0.3_t0.27_0.32_t750_k0.0008_10.sdds')
        assert result == expected
    
    @patch('src.tracking.DATA_DIR', Path('/data'))
    def test_get_tbt_path_negative_index(self):
        """Test TBT path generation with negative index."""
        result = get_tbt_path(beam=1, nturns=100, index=-1)
        expected = Path('/data/tbt_b1_c0_t0.28_0.31_t100_k0.001_-1.sdds')
        assert result == expected


class TestCorrectOrbit:
    """Test suite for correct_orbit function."""
    
    def test_correct_orbit_default_deltap(self):
        """Test orbit correction with default deltap value."""
        mock_mad = Mock()
        mock_mad.send.return_value = mock_mad
        
        with patch('src.tracking.match_tunes') as mock_match_tunes:
            correct_orbit(mock_mad, beam=1, tunes=[0.28, 0.31])
            
            # Verify MAD commands were sent
            mock_mad.send.assert_called_once()
            sent_command = mock_mad.send.call_args[0][0]
            assert "local deltap = nil;" in sent_command
            assert "sequence = MADX.lhcb1" in sent_command
            assert "method=\"svd\"" in sent_command
            
            # Verify tune matching was called
            mock_match_tunes.assert_called_once_with(mock_mad, 1, [0.28, 0.31], deltap="nil")
    
    def test_correct_orbit_beam2_custom_deltap(self):
        """Test orbit correction for beam 2 with custom deltap."""
        mock_mad = Mock()
        mock_mad.send.return_value = mock_mad
        
        with patch('src.tracking.match_tunes') as mock_match_tunes:
            correct_orbit(mock_mad, beam=2, tunes=[0.30, 0.32], deltap=0.001)
            
            # Verify MAD commands were sent with correct beam and deltap
            sent_command = mock_mad.send.call_args[0][0]
            assert "local deltap = 0.001;" in sent_command
            assert "sequence = MADX.lhcb2" in sent_command
            
            # Verify tune matching was called with correct parameters
            mock_match_tunes.assert_called_once_with(mock_mad, 2, [0.30, 0.32], deltap=0.001)
    
    def test_correct_orbit_string_deltap(self):
        """Test orbit correction with string deltap value."""
        mock_mad = Mock()
        mock_mad.send.return_value = mock_mad
        
        with patch('src.tracking.match_tunes') as mock_match_tunes:
            correct_orbit(mock_mad, beam=1, tunes=[0.25, 0.35], deltap="0.005")
            
            sent_command = mock_mad.send.call_args[0][0]
            assert "local deltap = 0.005;" in sent_command
            
            mock_match_tunes.assert_called_once_with(mock_mad, 1, [0.25, 0.35], deltap="0.005")
    
    def test_correct_orbit_logging(self, caplog):
        """Test that orbit correction logs appropriate messages."""
        mock_mad = Mock()
        mock_mad.send.return_value = mock_mad
        
        with patch('src.tracking.match_tunes'), caplog.at_level(logging.INFO):
            correct_orbit(mock_mad, beam=2, tunes=[0.28, 0.31], deltap=0.002)
            
            # Check that both log messages were recorded
            assert len(caplog.records) == 2
            assert "Orbit correction completed for beam 2 with deltap = 0.002" in caplog.text
            assert "Tune matching completed for beam 2 with tunes = [0.28, 0.31]" in caplog.text


class TestRunTracking:
    """Test suite for run_tracking function."""
    
    def test_run_tracking_successful_execution(self):
        """Test successful tracking simulation execution."""
        # Mock the MAD context manager
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        # Setup MAD instance behavior
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        
        # Create a sample DataFrame
        sample_df = pd.DataFrame({
            'name': ['BPM1', 'BPM2'],
            'x': [0.001, 0.002],
            'y': [0.0015, 0.0025],
            'eidx': [1, 2],
            'turn': [1, 1],
            'id': [1, 2]
        })
        mock_mad_instance.mtbl.to_df.return_value = sample_df
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng') as mock_start_madng, \
             patch('src.tracking.observe_BPMs') as mock_observe_bpms:
            
            result = run_tracking(
                beam=1, 
                nturns=1000, 
                model_dir=Path('/test/model')
            )
            
            # Verify MAD initialization functions were called
            mock_start_madng.assert_called_once_with(
                mock_mad_instance, 1, Path('/test/model'), tunes=[0.28, 0.31]
            )
            mock_observe_bpms.assert_called_once_with(mock_mad_instance, 1)
            
            # Verify tracking command was sent
            mock_mad_instance.send.assert_called()
            tracking_command = mock_mad_instance.send.call_args[0][0]
            assert "track {sequence = MADX.lhcb1, nturn = 1000" in tracking_command
            assert "deltap = nil" in tracking_command
            
            # Verify kick amplitude was sent
            mock_mad_instance.send().send.assert_called_once_with(1e-3)
            
            # Verify DataFrame retrieval
            mock_mad_instance.mtbl.to_df.assert_called_once_with(
                columns=["name", "x", "y", "eidx", "turn", "id"]
            )
            
            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert list(result.columns) == ['name', 'x', 'y', 'eidx', 'turn', 'id']
    
    def test_run_tracking_beam2_custom_params(self):
        """Test tracking simulation for beam 2 with custom parameters."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng') as mock_start_madng, \
             patch('src.tracking.observe_BPMs') as mock_observe_bpms:
            
            run_tracking(
                beam=2, 
                nturns=500, 
                model_dir=Path('/custom/model'),
                tunes=[0.30, 0.32],
                kick_amp=2e-3,
                deltap=0.001
            )
            
            # Verify correct parameters were passed
            mock_start_madng.assert_called_once_with(
                mock_mad_instance, 2, Path('/custom/model'), tunes=[0.30, 0.32]
            )
            mock_observe_bpms.assert_called_once_with(mock_mad_instance, 2)
            
            # Verify tracking command includes custom parameters
            tracking_command = mock_mad_instance.send.call_args[0][0]
            assert "track {sequence = MADX.lhcb2, nturn = 500" in tracking_command
            assert "deltap = 0.001" in tracking_command
            
            # Verify custom kick amplitude was sent
            mock_mad_instance.send().send.assert_called_once_with(2e-3)
    
    def test_run_tracking_with_orbit_correction(self):
        """Test tracking simulation with orbit correction when deltap is not nil."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'), \
             patch('src.tracking.correct_orbit') as mock_correct_orbit:
            
            run_tracking(
                beam=1, 
                nturns=100, 
                model_dir=Path('/test/model'),
                deltap=0.002
            )
            
            # Verify orbit correction was called
            mock_correct_orbit.assert_called_once_with(
                mock_mad_instance, 1, [0.28, 0.31], deltap=0.002
            )
    
    def test_run_tracking_no_orbit_correction_with_nil_deltap(self):
        """Test that orbit correction is not called when deltap is nil."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'), \
             patch('src.tracking.correct_orbit') as mock_correct_orbit:
            
            run_tracking(
                beam=1, 
                nturns=100, 
                model_dir=Path('/test/model'),
                deltap="nil"
            )
            
            # Verify orbit correction was NOT called
            mock_correct_orbit.assert_not_called()
    
    def test_run_tracking_logging(self, caplog):
        """Test that tracking completion is logged."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'), \
             caplog.at_level(logging.INFO):
            
            run_tracking(beam=2, nturns=250, model_dir=Path('/test/model'))
            
            # Verify completion message was logged
            assert "Tracking complete for beam 2 over 250 turns." in caplog.text
    
    def test_run_tracking_mad_exception_handling(self):
        """Test that exceptions from MAD operations are properly handled."""
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(side_effect=Exception("MAD initialization failed"))
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             pytest.raises(Exception, match="MAD initialization failed"):
            run_tracking(beam=1, nturns=100, model_dir=Path('/test/model'))
    
    def test_run_tracking_beta_function_calculation(self):
        """Test that beta function calculation is included in tracking command."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'):
            
            run_tracking(beam=1, nturns=100, model_dir=Path('/test/model'))
            
            # Verify beta function calculation is in the tracking command
            tracking_command = mock_mad_instance.send.call_args[0][0]
            assert 'local betx = tws["beta11"][1]' in tracking_command
            assert 'local bety = tws["beta22"][1]' in tracking_command
            assert 'local sqrt_betx = math.sqrt(betx)' in tracking_command
            assert 'local sqrt_bety = math.sqrt(bety)' in tracking_command
            assert 'x = kick_amp * sqrt_betx, y = kick_amp * sqrt_bety' in tracking_command
    
    def test_run_tracking_zero_turns(self):
        """Test tracking simulation with zero turns."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'):
            
            result = run_tracking(beam=1, nturns=0, model_dir=Path('/test/model'))
            
            # Verify tracking command includes zero turns
            tracking_command = mock_mad_instance.send.call_args[0][0]
            assert "nturn = 0" in tracking_command
            
            assert isinstance(result, pd.DataFrame)
    
    def test_run_tracking_extreme_kick_amplitude(self):
        """Test tracking simulation with extreme kick amplitude values."""
        mock_mad_instance = Mock()
        mock_mad_context = Mock()
        mock_mad_context.__enter__ = Mock(return_value=mock_mad_instance)
        mock_mad_context.__exit__ = Mock(return_value=None)
        
        mock_mad_instance.send.return_value = mock_mad_instance
        mock_mad_instance.mtbl = Mock()
        mock_mad_instance.mtbl.to_df.return_value = pd.DataFrame()
        
        with patch('src.tracking.MAD', return_value=mock_mad_context), \
             patch('src.tracking.start_madng'), \
             patch('src.tracking.observe_BPMs'):
            
            # Test with very small kick amplitude
            run_tracking(beam=1, nturns=10, model_dir=Path('/test/model'), kick_amp=1e-9)
            mock_mad_instance.send().send.assert_called_with(1e-9)
            
            # Test with very large kick amplitude
            run_tracking(beam=2, nturns=10, model_dir=Path('/test/model'), kick_amp=1.0)
            mock_mad_instance.send().send.assert_called_with(1.0)


# Integration tests
class TestTrackingIntegration:
    """Integration tests for tracking module functions."""
    
    def test_path_functions_consistency(self):
        """Test that TFS and TBT path functions use consistent suffixes."""
        beam = 1
        nturns = 1000
        coupling_knob = 0.5
        tunes = [0.29, 0.33]
        kick_amp = 2e-3
        
        # Get paths
        with patch('src.tracking.DATA_DIR', Path('/data')):
            tfs_path = get_tfs_path(beam, nturns, coupling_knob, tunes, kick_amp)
            tbt_path = get_tbt_path(beam, nturns, coupling_knob, tunes, kick_amp, index=0)
        
        # Extract suffixes from paths
        tfs_suffix = tfs_path.stem.replace('.tfs', '')  # Remove .tfs from .tfs.bz2
        tbt_suffix = tbt_path.stem.replace('tbt_', '').replace('_0', '')  # Remove tbt_ prefix and _0 index
        
        assert tfs_suffix == tbt_suffix
    
    def test_file_suffix_used_in_paths(self):
        """Test that get_file_suffix output is correctly used in path functions."""
        beam = 2
        nturns = 500
        
        suffix = get_file_suffix(beam, nturns)
        
        with patch('src.tracking.DATA_DIR', Path('/test')):
            tfs_path = get_tfs_path(beam, nturns)
            tbt_path = get_tbt_path(beam, nturns, index=0)
        
        assert suffix in str(tfs_path)
        assert suffix in str(tbt_path)


# Edge cases and error conditions
class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""
    
    def test_get_file_suffix_with_special_float_values(self):
        """Test file suffix generation with special float values."""
        # Test with infinity
        result = get_file_suffix(beam=1, nturns=100, kick_amp=float('inf'))
        assert "kinf" in result
        
        # Test with very small number
        result = get_file_suffix(beam=1, nturns=100, kick_amp=1e-15)
        assert "k1e-15" in result
    
    def test_get_file_suffix_with_empty_tunes_list(self):
        """Test file suffix generation with empty tunes list."""
        # This should not raise an error but should handle gracefully
        result = get_file_suffix(beam=1, nturns=100, tunes=[])
        assert isinstance(result, str)
    
    def test_get_file_suffix_with_single_tune(self):
        """Test file suffix generation with single tune value."""
        result = get_file_suffix(beam=1, nturns=100, tunes=[0.28])
        assert "t0.28" in result
    
    def test_path_functions_with_nonexistent_data_dir(self):
        """Test path functions behavior with non-existent DATA_DIR."""
        with patch('src.tracking.DATA_DIR', Path('/nonexistent/path')):
            tfs_path = get_tfs_path(beam=1, nturns=100)
            tbt_path = get_tbt_path(beam=1, nturns=100)
            
            # Paths should still be created even if directory doesn't exist
            assert isinstance(tfs_path, Path)
            assert isinstance(tbt_path, Path)
            assert "/nonexistent/path" in str(tfs_path)
            assert "/nonexistent/path" in str(tbt_path)


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests."""
    
    def test_get_file_suffix_performance_with_large_numbers(self):
        """Test file suffix generation performance with large numbers."""
        import time
        
        start_time = time.time()
        for i in range(1000):
            get_file_suffix(beam=1, nturns=i * 1000000, kick_amp=i * 1e-6)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
    
    def test_path_functions_with_many_calls(self):
        """Test path functions with many successive calls."""
        with patch('src.tracking.DATA_DIR', Path('/data')):
            paths = []
            for i in range(100):
                tfs_path = get_tfs_path(beam=1, nturns=i)
                tbt_path = get_tbt_path(beam=2, nturns=i, index=i)
                paths.extend([tfs_path, tbt_path])
            
            # All paths should be unique
            assert len(set(str(p) for p in paths)) == len(paths)


if __name__ == "__main__":
    pytest.main([__file__])