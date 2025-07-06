import pytest
from hello import main


def test_main_prints_correct_message(capsys):
    """Test that main function prints the expected message."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from ai-batch!\n"
    assert captured.err == ""


def test_main_returns_none():
    """Test that main function returns None."""
    result = main()
    assert result is None


def test_main_can_be_called_multiple_times(capsys):
    """Test that main function can be called multiple times consistently."""
    main()
    first_output = capsys.readouterr().out
    
    main()
    second_output = capsys.readouterr().out
    
    assert first_output == second_output == "Hello from ai-batch!\n"