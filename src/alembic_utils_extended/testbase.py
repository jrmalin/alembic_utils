# pylint: disable=unsupported-assignment-operation
import contextlib
import os
from io import StringIO
from pathlib import Path
from typing import Any, Callable, NoReturn

from alembic import command as alem_command
from alembic.config import Config
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

REPO_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.resolve()
TEST_RESOURCE_ROOT = REPO_ROOT / "src" / "test" / "resources"
TEST_VERSIONS_ROOT = REPO_ROOT / "src" / "test" / "alembic_config" / "versions"


ALEMBIC_COMMAND_MAP: dict[str, Callable[..., NoReturn]] = {
    "upgrade": alem_command.upgrade,
    "downgrade": alem_command.downgrade,
    "revision": alem_command.revision,
    "current": alem_command.current,
    "check": alem_command.check,
}


def build_alembic_config(engine: Engine) -> Config:
    """Populate alembic configuration from metadata and config file."""
    path_to_alembic_ini = REPO_ROOT / "alembic.ini"

    alembic_cfg = Config(path_to_alembic_ini)

    alembic_cfg.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False))

    alembic_cfg.set_main_option("script_location", str((Path("src") / "test" / "alembic_config")))
    return alembic_cfg


def run_alembic_command(
    engine: Engine,
    command: str,
    command_kwargs: dict[str, Any],
    target_metadata: MetaData | None = None,
    compare_check_constraints: bool = False,
    compare_enum_values: bool = False,
    ignore_enum_label_removal: set[str] | None = None,
    compare_indexes: bool = False,
    compare_tables: bool = False,
) -> str:
    command_func = ALEMBIC_COMMAND_MAP[command]

    stdout = StringIO()

    alembic_cfg = build_alembic_config(engine)
    with engine.begin() as connection:
        alembic_cfg.attributes["connection"] = connection
        if target_metadata is not None:
            alembic_cfg.attributes["target_metadata"] = target_metadata
        if compare_check_constraints:
            alembic_cfg.attributes["compare_check_constraints"] = compare_check_constraints
        if compare_enum_values:
            alembic_cfg.attributes["compare_enum_values"] = compare_enum_values
        if ignore_enum_label_removal:
            alembic_cfg.attributes["ignore_enum_label_removal"] = ignore_enum_label_removal
        if compare_indexes:
            alembic_cfg.attributes["compare_indexes"] = compare_indexes
        if compare_tables:
            alembic_cfg.attributes["compare_tables"] = compare_tables
        with contextlib.redirect_stdout(stdout):
            command_func(alembic_cfg, **command_kwargs)
    return stdout.getvalue()
