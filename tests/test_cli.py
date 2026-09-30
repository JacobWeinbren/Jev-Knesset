"""The cli step list."""
from knesset_ches import cli


def test_help_lists_every_step_and_an_unknown_step_is_refused(capsys):
    assert cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert all(f"\n    {step} " in help_text for step in cli.STEPS)
    assert cli.main(["draw"]) == 2
