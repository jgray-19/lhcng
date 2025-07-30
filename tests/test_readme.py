"""
Unit tests for README.md validation and content verification.

This test suite validates the README.md file structure, content,
and ensures it meets documentation standards for the lhcng package.

Testing Framework: unittest (Python standard library)
The tests focus on validating the actual README content which describes
lhcng as a package providing helpful functions for using MAD-NG with
accelerator models.
"""

import re
import unittest
from pathlib import Path


class TestReadmeValidation(unittest.TestCase):
    """Test suite for README.md file validation and basic structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.readme_path = Path("README.md")
        self.expected_content = """# lhcng
Helpful functions for using MAD-NG with accelerator models. By default the package
assumes the LHC layout but you can override this by setting the ``ACCEL``
environment variable to the desired machine name. The utilities are also useful
when used together with ``omc3``."""
        
    def test_readme_file_exists(self):
        """Test that README.md file exists in the project root."""
        self.assertTrue(
            self.readme_path.exists(),
            "README.md file should exist in the project root"
        )
        
    def test_readme_file_not_empty(self):
        """Test that README.md file is not empty."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertGreater(
                len(content.strip()),
                0,
                "README.md file should not be empty"
            )
            
    def test_readme_contains_project_title(self):
        """Test that README contains the project title 'lhcng'."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'lhcng',
                content.lower(),
                "README should contain the project name 'lhcng'"
            )
            
    def test_readme_has_correct_header(self):
        """Test that README starts with the correct markdown header."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.strip().split('\n')
            
            self.assertEqual(
                lines[0].strip(),
                '# lhcng',
                "README should start with '# lhcng' header"
            )
            
    def test_readme_describes_mad_ng_integration(self):
        """Test that README mentions MAD-NG integration."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'MAD-NG',
                content,
                "README should mention MAD-NG integration"
            )
            
    def test_readme_mentions_lhc_layout(self):
        """Test that README mentions LHC layout support."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'LHC',
                content,
                "README should mention LHC layout support"
            )
            
    def test_readme_mentions_accel_environment_variable(self):
        """Test that README mentions ACCEL environment variable correctly."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                '``ACCEL``',
                content,
                "README should mention ACCEL environment variable with proper formatting"
            )
            
    def test_readme_mentions_omc3_integration(self):
        """Test that README mentions omc3 integration."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                '``omc3``',
                content,
                "README should mention omc3 integration with proper formatting"
            )

    def test_readme_contains_helpful_functions_description(self):
        """Test that README describes helpful functions."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'Helpful functions',
                content,
                "README should describe helpful functions"
            )

    def test_readme_mentions_accelerator_models(self):
        """Test that README mentions accelerator models."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'accelerator models',
                content,
                "README should mention accelerator models"
            )

    def test_readme_describes_default_behavior(self):
        """Test that README explains default package behavior."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'By default the package',
                content,
                "README should explain default package behavior"
            )

    def test_readme_explains_override_mechanism(self):
        """Test that README explains how to override defaults."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'you can override this by setting',
                content,
                "README should explain how to override default settings"
            )

    def test_readme_mentions_machine_name_configuration(self):
        """Test that README mentions machine name configuration."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIn(
                'desired machine name',
                content,
                "README should mention machine name configuration"
            )


class TestReadmeFormatting(unittest.TestCase):
    """Test suite for README formatting and style consistency."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.readme_path = Path("README.md")

    def test_readme_file_encoding_is_utf8(self):
        """Test that README file is UTF-8 encoded."""
        if self.readme_path.exists():
            try:
                self.readme_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                self.fail("README.md should be UTF-8 encoded")

    def test_readme_line_length_reasonable(self):
        """Test that README lines are not excessively long."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            long_lines = [
                (i+1, line) for i, line in enumerate(lines) 
                if len(line) > 120 and not line.strip().startswith('http')
            ]
            
            self.assertEqual(
                len(long_lines),
                0,
                f"README should not have excessively long lines (>120 chars): {long_lines}"
            )

    def test_readme_no_trailing_whitespace(self):
        """Test that README lines don't have trailing whitespace."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            lines_with_trailing_space = [
                i + 1 for i, line in enumerate(lines)
                if line.rstrip() != line and line.strip() != ''
            ]
            
            self.assertEqual(
                len(lines_with_trailing_space),
                0,
                f"README should not have trailing whitespace on lines: {lines_with_trailing_space}"
            )

    def test_readme_proper_code_formatting(self):
        """Test that code elements are properly formatted with backticks."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Check that ACCEL and omc3 are properly formatted
            self.assertIn('``ACCEL``', content, "ACCEL should be formatted with double backticks")
            self.assertIn('``omc3``', content, "omc3 should be formatted with double backticks")

    def test_readme_consistent_paragraph_structure(self):
        """Test that README has consistent paragraph structure."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Should not have excessive blank lines
            consecutive_blank_lines = 0
            max_consecutive_blank = 0
            
            for line in lines:
                if line.strip() == '':
                    consecutive_blank_lines += 1
                    max_consecutive_blank = max(max_consecutive_blank, consecutive_blank_lines)
                else:
                    consecutive_blank_lines = 0
            
            self.assertLessEqual(
                max_consecutive_blank,
                2,
                "README should not have more than 2 consecutive blank lines"
            )

    def test_readme_consistent_newlines(self):
        """Test that README uses consistent line endings."""
        if self.readme_path.exists():
            with open(self.readme_path, 'rb') as f:
                raw_content = f.read()
            
            # Check for consistent line endings (prefer Unix-style \n)
            crlf_count = raw_content.count(b'\r\n')
            lf_count = raw_content.count(b'\n') - crlf_count
            cr_count = raw_content.count(b'\r') - crlf_count
            
            # Should primarily use one type of line ending
            total_lines = crlf_count + lf_count + cr_count
            if total_lines > 0:
                dominant_type_count = max(crlf_count, lf_count, cr_count)
                consistency_ratio = dominant_type_count / total_lines
                
                self.assertGreaterEqual(
                    consistency_ratio,
                    0.9,
                    "README should use consistent line endings"
                )


class TestReadmeContentIntegrity(unittest.TestCase):
    """Test suite for README content integrity and completeness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.readme_path = Path("README.md") 
        
    def test_readme_contains_all_key_components(self):
        """Test that README contains all essential components."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Essential components based on actual content
            essential_components = [
                'lhcng',           # Project name
                'Helpful functions', # Purpose description
                'MAD-NG',          # Primary integration
                'accelerator models', # Domain
                'By default',      # Default behavior explanation
                'LHC layout',      # Default configuration
                'override',        # Configuration capability
                'ACCEL',           # Environment variable
                'environment variable', # Configuration method
                'machine name',    # Configuration target
                'utilities',       # Additional description
                'omc3'            # Secondary integration
            ]
            
            missing_components = [
                component for component in essential_components
                if component not in content
            ]
            
            self.assertEqual(
                len(missing_components),
                0,
                f"README should contain these essential components: {missing_components}"
            )

    def test_readme_provides_complete_usage_context(self):
        """Test that README provides comprehensive usage context."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            usage_contexts = [
                'for using',       # Purpose context
                'By default',      # Default behavior
                'you can override', # Customization
                'also useful',     # Additional use cases
                'when used together' # Integration context
            ]
            
            found_contexts = [
                context for context in usage_contexts
                if context in content
            ]
            
            self.assertGreaterEqual(
                len(found_contexts),
                4,
                f"README should provide comprehensive usage context. Found: {found_contexts}"
            )

    def test_readme_technical_terminology_consistency(self):
        """Test that technical terms are used consistently."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Check MAD-NG consistency
            mad_variants = re.findall(r'MAD[-\s]?NG', content, re.IGNORECASE)
            if mad_variants:
                consistent_variants = [v for v in mad_variants if v == 'MAD-NG']
                consistency_ratio = len(consistent_variants) / len(mad_variants)
                
                self.assertGreaterEqual(
                    consistency_ratio,
                    1.0,
                    "MAD-NG should be consistently hyphenated"
                )

    def test_readme_sentence_structure_quality(self):
        """Test that README has proper sentence structure."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Remove the header for sentence analysis
            content_without_header = '\n'.join(content.split('\n')[1:])
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', content_without_header)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Should have multiple complete sentences
            self.assertGreaterEqual(
                len(sentences),
                3,
                "README should contain multiple complete sentences"
            )
            
            # Check sentence capitalization
            capitalized_sentences = [
                s for s in sentences
                if s and s[0].isupper()
            ]
            
            capitalization_ratio = len(capitalized_sentences) / max(len(sentences), 1)
            self.assertGreaterEqual(
                capitalization_ratio,
                0.8,
                "Most sentences in README should start with capital letters"
            )

    def test_readme_describes_package_purpose_clearly(self):
        """Test that README clearly describes what the package does."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Should contain clear action words and purpose
            purpose_indicators = [
                'functions',       # What it provides
                'for using',       # Purpose preposition
                'with',           # Integration indicator
                'useful'          # Value proposition
            ]
            
            found_indicators = sum(
                1 for indicator in purpose_indicators
                if indicator in content.lower()
            )
            
            self.assertGreaterEqual(
                found_indicators,
                3,
                "README should clearly describe package purpose"
            )


class TestReadmeEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for README validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.readme_path = Path("README.md")
    
    def test_readme_content_not_none(self):
        """Test that README content is never None."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            self.assertIsNotNone(
                content,
                "README content should never be None"
            )
            
    def test_readme_handles_line_processing(self):
        """Test that README can handle line-by-line processing."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Should be able to process lines without errors
            processed_lines = [line.strip() for line in lines]
            self.assertIsInstance(
                processed_lines,
                list,
                "Should be able to process README lines as a list"
            )
            
            # Should handle empty and non-empty lines
            non_empty_lines = [line for line in processed_lines if line]
            self.assertGreater(
                len(non_empty_lines),
                0,
                "Should have at least one non-empty line"
            )

    def test_readme_no_problematic_characters(self):
        """Test that README doesn't contain problematic control characters."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Should not contain problematic characters
            problematic_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
            found_problematic = [
                char for char in problematic_chars
                if char in content
            ]
            
            self.assertEqual(
                len(found_problematic),
                0,
                f"README should not contain problematic control characters: {found_problematic}"
            )

    def test_readme_word_count_appropriate(self):
        """Test that README has an appropriate word count."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Count words (excluding header)
            content_without_header = '\n'.join(content.split('\n')[1:])
            words = content_without_header.split()
            word_count = len(words)
            
            # Should have enough content to be informative but not too verbose
            self.assertGreaterEqual(
                word_count,
                25,
                "README should have sufficient content (at least 25 words)"
            )
            
            self.assertLessEqual(
                word_count,
                200,
                "README should be concise (less than 200 words)"
            )

    def test_readme_structural_integrity(self):
        """Test that README has proper structural integrity."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            lines = content.strip().split('\n')
            
            # Should have header + content
            self.assertGreaterEqual(
                len(lines),
                2,
                "README should have at least a header and content"
            )
            
            # First line should be header
            self.assertTrue(
                lines[0].startswith('#'),
                "First line should be a markdown header"
            )
            
            # Should not be just a header
            content_lines = [line for line in lines[1:] if line.strip()]
            self.assertGreater(
                len(content_lines),
                0,
                "README should have content beyond just the header"
            )

    def test_readme_character_set_validity(self):
        """Test that README uses valid character sets."""
        if self.readme_path.exists():
            content = self.readme_path.read_text(encoding='utf-8')
            
            # Should be printable or whitespace
            non_printable_chars = [
                char for char in content
                if not (char.isprintable() or char.isspace())
            ]
            
            self.assertEqual(
                len(non_printable_chars),
                0,
                f"README should only contain printable characters. Found: {non_printable_chars}"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)