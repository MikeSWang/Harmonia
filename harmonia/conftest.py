"""Test configuration for :mod:`harmonia`.

"""
import pytest


def pytest_addoption(parser):
    """Add command-line options to `pytest` parser.

    Parameters
    ----------
    parser : :class:`_pytest.config.argparsing.Parser`
        `pytest` parser object.

    """
    parser.addoption(
        '--runslow', action='store_true', default=False, help="Run slow tests."
    )


def pytest_configure(config):
    """Add ini-file options to `pytest` configuration.

    Parameters
    ----------
    config : :class:`_pytest.config.Config`
        `pytest` configuration object.

    """
    config.addinivalue_line('markers', "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Modify test collection items.

    Parameters
    ----------
    config : :class:`_pytest.config.Config`
        `pytest` configuration object.
    items : list of :class:`_pytest.nodes.Item`
        `pytest` item objects.

    """
    if config.getoption('--runslow'):
        return

    skip_slow = pytest.mark.skip(
        reason="Use --runslow option to run slow tests."
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
