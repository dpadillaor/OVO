import pytest
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from ovo.entities.logger import Logger

@pytest.fixture
def logger(tmp_path):
    # Setup a logger with a temp directory
    return Logger(output_path=str(tmp_path), pid=None, use_wandb=False)

def test_start_session_records_time(logger):
    logger.start_session()
    assert logger.start_time > 0
    assert logger.start_time <= time.time()

def test_record_step_accumulates_data(logger):
    logger.record_step(0.5)
    logger.record_step(1.5)
    assert logger.stats["spf"] == [0.5, 1.5]

@patch("ovo.entities.logger.Logger.log_memory_usage")
def test_record_frame_end_capsules_logic(mock_mem, logger):
    logger.record_frame_end(frame_id=10, duration=0.2)
    
    # Should have called memory log
    mock_mem.assert_called_once_with(10)
    # Should have recorded spf
    assert logger.stats["spf"] == [0.2]

def test_finalize_calculates_fps_correctly(logger):
    logger.start_time = time.time() - 10.0 # 10 seconds ago
    
    # Processed 100 items in 10 seconds -> 10 FPS
    logger.finalize(total_items=100)
    
    # Allow small delta due to time.time() call in finalize
    assert 9.9 <= logger.stats["avg_fps"][0] <= 10.1

@patch("ovo.entities.logger.Logger.write_stats")
@patch("ovo.entities.logger.Logger.print_final_stats")
def test_finalize_triggers_io_and_reports(mock_print, mock_write, logger):
    logger.finalize(total_items=10)
    
    mock_write.assert_called_once()
    mock_print.assert_called_once()
