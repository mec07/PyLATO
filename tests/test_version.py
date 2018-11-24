import re


def test_version_numbers():
    # check that the change log has an entry for this version
    changelog = None
    with open('CHANGELOG.md', 'r') as fh:
        match = re.search(
            "## \[([0-9]+.[0-9]+.[0-9]+)\]", fh.read())
        if match:
            changelog = match.group(1)
            print('CHANGELOG.md version: {}'.format(changelog))
        else:
            print('CHANGELOG.md version MISSING!')

    # check the setup.py version number
    setup = None
    with open('setup.py', 'r') as fh:
        match = re.search(r"\s*version\s*=\s*'([^']+)'", fh.read())
        if match:
            setup = match.group(1)
            print('setup.py version: {}'.format(setup))
        else:
            print('setup.py version MISSING!')

    assert changelog is not None
    assert setup == changelog
