from behave.fixture import use_fixture_by_tag, fixture_call_params
from features.support.fixtures import backup_file

# -- REGISTRY DATA SCHEMA: (fixture_func, fixture_args, fixture_kwargs)
# Note: See https://behave.readthedocs.io/en/latest/fixtures.html for more info
# on fixtures and how they work
fixture_registry = {
    "fixture.backup.TBcanonical_s.json": fixture_call_params(
        backup_file, filename="models/TBcanonical_s.json"),
}


def before_tag(context, tag):
    if tag.startswith("fixture."):
        return use_fixture_by_tag(tag, context, fixture_registry)


def before_scenario(context, scenario):
    context.patchers = []


def after_scenario(context, scenario):
    for patcher in context.patchers:
        patcher.stop()
