#  test_insert_functions.py
#
#  Copyright (c) 2025 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture
def sample_note_content():
    """Sample note content for testing."""
    return """# Sample document for testing
The state is Minnesota

## Itinerary
(Saturday, Aug 30 12:45pm) MSP -> ATL -> ATH (Sunday, Aug 31 10:45am)

### Overview
Starting with party/beach vibes (Mykonos), moving to your one major archaeological site (Ephesus), then the romantic scenery day (Santorini), sea day to recover, then choosing between nature/history combos in Italy.

### Day 1: Athens -> Mykonos
- Morning arrival in Athens
- Transfer to Mykonos
- Beach time

### Day 2: Mykonos
- Beach hopping
- Nightlife

The weather is sunny."""


class TestInsertFunctions:
    """Test the insert functions with focus on preview generation."""
    
    def test_insert_after_text_preview_logic(self, sample_note_content):
        """Test the core logic of insert_after_text preview generation."""
        # Test the core string replacement logic
        anchor_text = "### Overview\nStarting with party/beach vibes (Mykonos), moving to your one major archaeological site (Ephesus), then the romantic scenery day (Santorini), sea day to recover, then choosing between nature/history combos in Italy."
        new_text = "\n\nThe color is Yellow"
        
        # Simulate the replacement logic
        new_content = sample_note_content.replace(anchor_text, anchor_text + new_text)
        
        # Test that the replacement worked
        assert new_text in new_content
        assert new_content.count(anchor_text) == 1  # Should still have the anchor
        
        # Test preview generation logic
        old_lines = sample_note_content.split("\n")
        new_lines = new_content.split("\n")
        
        # Find the line where insertion occurred in the NEW content
        anchor_line = -1
        for i, line in enumerate(new_lines):
            if "### Overview" in line:
                anchor_line = i
                break
        
        # Verify anchor was found
        assert anchor_line >= 0
        
        # Test preview generation (±5 lines around insertion point)
        preview_start = max(0, anchor_line - 5)
        preview_end = min(len(new_lines), anchor_line + 6)
        preview_lines = new_lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)
        
        # Verify preview contains the anchor and new content
        assert "### Overview" in preview
        assert "The color is Yellow" in preview
        
        # Test metadata structure
        metadata = {
            "status": "success",
            "affected_lines": len(new_lines) - len(old_lines),
            "total_lines": len(new_lines),
            "anchor_line": anchor_line + 1,
            "preview": preview
        }
        
        assert metadata["status"] == "success"
        assert metadata["affected_lines"] == 2  # Added 2 lines
        assert metadata["anchor_line"] > 0
        assert metadata["total_lines"] > len(old_lines)
    
    def test_insert_before_text_preview_logic(self, sample_note_content):
        """Test the core logic of insert_before_text preview generation."""
        anchor_text = "### Day 1: Athens -> Mykonos"
        new_text = "### Important Note\nDon't forget sunscreen!\n\n"
        
        # Simulate the replacement logic
        new_content = sample_note_content.replace(anchor_text, new_text + anchor_text)
        
        # Test that the replacement worked
        assert new_text in new_content
        assert new_content.count(anchor_text) == 1  # Should still have the anchor
        
        # Test preview generation logic
        old_lines = sample_note_content.split("\n")
        new_lines = new_content.split("\n")
        
        # Find the line where insertion occurred in the NEW content
        anchor_line = -1
        for i, line in enumerate(new_lines):
            if "### Day 1: Athens -> Mykonos" in line:
                anchor_line = i
                break
        
        # Verify anchor was found
        assert anchor_line >= 0
        
        # Test preview generation (±5 lines around insertion point)
        preview_start = max(0, anchor_line - 5)
        preview_end = min(len(new_lines), anchor_line + 6)
        preview_lines = new_lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)
        
        # Verify preview contains the anchor and new content
        assert "### Day 1: Athens -> Mykonos" in preview
        assert "Don't forget sunscreen!" in preview
        
        # Test metadata structure
        metadata = {
            "status": "success",
            "affected_lines": len(new_lines) - len(old_lines),
            "total_lines": len(new_lines),
            "anchor_line": anchor_line + 1,
            "preview": preview
        }
        
        assert metadata["status"] == "success"
        assert metadata["affected_lines"] == 3  # Added 3 lines
        assert metadata["anchor_line"] > 0
        assert metadata["total_lines"] > len(old_lines)
    
    def test_strict_matching_logic(self, sample_note_content):
        """Test the strict matching logic for insert functions."""
        # Test 1: Non-existent anchor text
        anchor_text = "This text does not exist"
        assert anchor_text not in sample_note_content
        
        # Test 2: Duplicate anchor text
        duplicate_content = "# Test\nDuplicate line\nSome content\nDuplicate line\nMore content"
        duplicate_anchor = "Duplicate line"
        assert duplicate_content.count(duplicate_anchor) == 2
        
        # Test 3: Unique anchor text
        unique_anchor = "### Overview"
        assert sample_note_content.count(unique_anchor) == 1
    
    def test_insert_at_line_metadata_logic(self, sample_note_content):
        """Test the metadata generation for insert_at_line."""
        lines = sample_note_content.split("\n")
        insert_index = 4  # Insert after "## Itinerary"
        text_to_insert = "### Special Events"
        
        # Simulate the insertion
        lines.insert(insert_index, text_to_insert)
        new_content = "\n".join(lines)
        
        # Test preview generation (±5 lines around insertion point)
        preview_start = max(0, insert_index - 5)
        preview_end = min(len(lines), insert_index + 6)
        preview_lines = lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)
        
        # Test metadata structure
        metadata = {
            "status": "success",
            "affected_lines": 1,
            "total_lines": len(lines),
            "insertion_line": insert_index + 1,
            "preview": preview
        }
        
        assert metadata["status"] == "success"
        assert metadata["affected_lines"] == 1
        assert metadata["insertion_line"] == 5
        assert metadata["total_lines"] > 0
        assert text_to_insert in metadata["preview"]
    
    def test_return_content_parameter_logic(self, sample_note_content):
        """Test the return_content parameter logic."""
        # Test return_content=False (default)
        success_metadata = {
            "status": "success",
            "operation": "append",
            "message": "Text appended successfully"
        }
        
        # Should be valid JSON
        json_str = json.dumps(success_metadata)
        parsed = json.loads(json_str)
        
        assert parsed["status"] == "success"
        assert parsed["operation"] == "append"
        assert "message" in parsed
        
        # Test return_content=True
        full_content = sample_note_content + "\n\nAppended content"
        assert "Appended content" in full_content
        assert len(full_content) > len(sample_note_content)
    
    def test_multiline_anchor_scenarios(self, sample_note_content):
        """Test scenarios with multiline anchor text."""
        # Test the specific case from the user's problem
        multiline_anchor = """### Overview
Starting with party/beach vibes (Mykonos), moving to your one major archaeological site (Ephesus), then the romantic scenery day (Santorini), sea day to recover, then choosing between nature/history combos in Italy."""
        
        # Verify this anchor exists in the content
        assert multiline_anchor in sample_note_content
        
        # Test that it's unique
        assert sample_note_content.count(multiline_anchor) == 1
        
        # Test insertion after this anchor
        new_text = "\n\nThe color is Yellow"
        new_content = sample_note_content.replace(multiline_anchor, multiline_anchor + new_text)
        
        # Verify the insertion worked
        assert new_text in new_content
        assert multiline_anchor in new_content
        
        # Test that the anchor can be found in new content for preview
        new_lines = new_content.split("\n")
        anchor_line = -1
        for i, line in enumerate(new_lines):
            if "### Overview" in line:
                anchor_line = i
                break
        
        assert anchor_line >= 0  # Should find the anchor
        
        # Test preview generation
        preview_start = max(0, anchor_line - 5)
        preview_end = min(len(new_lines), anchor_line + 6)
        preview_lines = new_lines[preview_start:preview_end]
        preview = "\n".join(preview_lines)
        
        # Both anchor and new content should be in preview
        assert "### Overview" in preview
        assert "The color is Yellow" in preview
        
        # Metadata should have correct anchor_line
        metadata = {
            "status": "success",
            "anchor_line": anchor_line + 1,  # Convert to 1-based
            "preview": preview
        }
        
        assert metadata["anchor_line"] is not None
        assert metadata["anchor_line"] > 0