"""Autogenerate support for native PostgreSQL enum types.

Alembic does not diff enum labels. Adding a member to a python ``Enum`` mapped with ``sa.Enum(..., native_enum=True)``
produces no migration at all, passes every other check, and then fails on the first write with
``invalid input value for enum``. This module closes that gap.

Opt in with ``context.configure(..., compare_enum_values=True)``, the same way ``pg_check_constraint`` uses
``compare_check_constraints``.

A label present in the database but absent from the models raises, since it usually means a python enum member was
deleted out from under live rows. PostgreSQL cannot drop an enum label, though, so schemas accumulate dead ones that
no migration can clear. Pass ``ignore_enum_label_removal={"some_type"}`` to tolerate them per type.

Scope is deliberately narrow. It emits ``ALTER TYPE ... ADD VALUE`` for labels the models declare and the database
lacks, and ``CREATE TYPE`` for a type that does not exist yet and will not be created implicitly. It never rebuilds a
type, never reorders existing labels, and never drops a type that is no longer declared -- all of those require
rewriting every dependent table under an ACCESS EXCLUSIVE lock, which is not something autogenerate should do behind
your back.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from alembic.autogenerate import comparators
from alembic.autogenerate.api import AutogenContext
from alembic.operations import ops
from sqlalchemy import ARRAY, Enum, MetaData

from alembic_utils_extended.exceptions import AlembicUtilsException
from alembic_utils_extended.pg_enum_ops import AddEnumValueOp, CreateEnumTypeOp

logger = logging.getLogger(__name__)


class EnumLabelRemovedError(AlembicUtilsException):
    """The database has an enum label the models no longer declare."""


class AmbiguousEnumDeclarationError(AlembicUtilsException):
    """Two columns declare the same enum type name with different labels."""


class MixedEnumTypeCreationError(AlembicUtilsException):
    """A new enum type is shared by tables that are being created and tables that already exist."""


@dataclass
class DeclaredEnum:
    """A native enum type as the models declare it."""

    schema: str | None
    name: str
    labels: tuple[str, ...]
    # Every (table, column) declaring this type. Drives both the error messages and the decision about whether
    # `CREATE TABLE` will create the type implicitly.
    declared_by: list[tuple[str, str]] = field(default_factory=list)

    @property
    def tables(self) -> set[str]:
        return {table for table, _ in self.declared_by}

    def describe_columns(self) -> str:
        return ", ".join(f"{table}.{column}" for table, column in self.declared_by)


def plan_enum_value_additions(
    type_name: str,
    declared: tuple[str, ...],
    defined: tuple[str, ...],
    *,
    schema: str | None = None,
) -> list[AddEnumValueOp]:
    """Plan the ``ALTER TYPE ... ADD VALUE`` statements that bring ``defined`` up to ``declared``.

    ``defined`` is consulted for *membership only*, never for order, and that is deliberate: the caller sources it
    from ``Inspector.get_enums``, which sorts labels by ``pg_enum.oid`` rather than ``pg_enum.enumsortorder``. Those
    agree only until someone runs a positional ``ADD VALUE``, after which the reflected order is simply wrong. Do not
    introduce a dependency on the order of ``defined`` here without changing how the caller reflects it.

    Each new label is anchored ``BEFORE`` the nearest following declared label that already exists, falling back to
    ``AFTER`` the nearest preceding one. Anchors are chosen only from labels actually present, so this still places
    labels sensibly when the database's overall order has drifted from the declaration order -- which is the normal
    case for any type whose history includes an unanchored ``ADD VALUE``.

    The running ``present`` set is what makes consecutive additions come out in the right order. PostgreSQL inserts
    immediately relative to the anchor, so repeated ``BEFORE 'X'`` stacks in emission order (``a`` then ``b`` gives
    ``a, b, X``) while repeated ``AFTER 'X'`` *reverses* (``X, b, a``). Marking each label present as it is planned
    means the second addition anchors on the first rather than on ``X``, so both directions preserve declared order.
    A stateless version that anchored off the original ``defined`` would silently reverse runs of appended labels.
    """
    present = set(defined)
    planned: list[AddEnumValueOp] = []
    for index, label in enumerate(declared):
        if label in present:
            continue

        following = next((candidate for candidate in declared[index + 1 :] if candidate in present), None)
        if following is not None:
            planned.append(AddEnumValueOp(type_name, label, before=following, schema=schema))
        else:
            preceding = next((candidate for candidate in reversed(declared[:index]) if candidate in present), None)
            # `preceding is None` means the type exists with no labels at all, which postgres allows only via
            # `CREATE TYPE t AS ENUM ()`. Appending unanchored is the only sensible move.
            planned.append(AddEnumValueOp(type_name, label, after=preceding, schema=schema))

        present.add(label)

    return planned


def _enum_labels(enum_type: Enum) -> tuple[str, ...]:
    """The labels sqlalchemy will send to postgres, with ``values_callable`` already applied."""
    return tuple(enum_type.enums)


def iter_native_enum_columns(metadata: MetaData | list[MetaData]) -> Iterator[tuple[str, str, Enum]]:
    """Yield ``(table name, column name, enum type)`` for every native enum column in the given metadata.

    ``native_enum`` defaults to ``True``, so a column that never mentions it is still native.
    """
    metadata_list = metadata if isinstance(metadata, list) else [metadata]
    for single_metadata in metadata_list:
        for table in single_metadata.tables.values():
            for column in table.columns:
                # An ARRAY of enums still depends on the underlying type.
                column_type = column.type.item_type if isinstance(column.type, ARRAY) else column.type
                if isinstance(column_type, Enum) and column_type.name and getattr(column_type, "native_enum", True):
                    yield table.name, column.name, column_type


def collect_declared_native_enums(metadata: MetaData | list[MetaData]) -> dict[tuple[str | None, str], DeclaredEnum]:
    """Every native enum type declared across the given metadata, keyed by ``(schema, type name)``.

    Types are deduplicated by name because copied columns -- sqlalchemy-continuum's version tables, for one --
    re-declare an independent ``Enum`` instance for the same underlying postgres type.
    """
    declared: dict[tuple[str | None, str], DeclaredEnum] = {}

    for table_name, column_name, enum_type in iter_native_enum_columns(metadata):
        key = (enum_type.schema, enum_type.name)
        labels = _enum_labels(enum_type)
        entry = declared.get(key)
        if entry is None:
            declared[key] = entry = DeclaredEnum(schema=enum_type.schema, name=enum_type.name, labels=labels)
        elif entry.labels != labels:
            raise AmbiguousEnumDeclarationError(
                f"Enum type {enum_type.name!r} is declared with conflicting labels: "
                f"{table_name}.{column_name} declares {list(labels)}, "
                f"but {entry.describe_columns()} declares {list(entry.labels)}."
            )

        entry.declared_by.append((table_name, column_name))

    return declared


@comparators.dispatch_for("schema")
def compare_enum_values(
    autogen_context: AutogenContext,
    upgrade_ops: ops.UpgradeOps,
    _schemas: list[str | None],
) -> None:
    if not autogen_context.opts.get("compare_enum_values") or autogen_context.dialect.name != "postgresql":
        return

    metadata = autogen_context.metadata
    if metadata is None:
        return

    declared = collect_declared_native_enums(metadata)
    if not declared:
        return

    inspector = autogen_context.inspector
    ignore_label_removal = set(autogen_context.opts.get("ignore_enum_label_removal") or ())
    default_schema = inspector.default_schema_name

    # Alias every default-schema type under `None` too, so an unqualified declaration resolves without a fallback.
    defined: dict[tuple[str | None, str], tuple[str, ...]] = {}
    for enum in inspector.get_enums(schema="*"):
        labels = tuple(enum["labels"])
        defined[(enum["schema"], enum["name"])] = labels
        if enum["schema"] == default_schema:
            defined.setdefault((None, enum["name"]), labels)

    # Only needed for types the database lacks, which is rare -- keep the reflection out of the common path.
    existing_tables: set[str] | None = None

    for key, declared_enum in sorted(declared.items(), key=lambda item: item[0][1]):
        defined_labels = defined.get(key)

        if defined_labels is None:
            if existing_tables is None:
                existing_tables = set(inspector.get_table_names(schema=default_schema))
            create_op = _plan_type_creation(declared_enum, existing_tables)
            if create_op is not None:
                logger.info("Detected CreateEnumTypeOp for %s", declared_enum.name)
                upgrade_ops.ops.append(create_op)
            continue

        orphans = [label for label in defined_labels if label not in declared_enum.labels]
        if orphans and declared_enum.name not in ignore_label_removal:
            raise EnumLabelRemovedError(
                f"Enum type {declared_enum.name!r} has label(s) {orphans} in the database that "
                f"{declared_enum.describe_columns()} no longer declare. PostgreSQL cannot remove an enum label, "
                "so this needs either the label(s) restored to the python enum, a full type rebuild, or the type "
                "listed in the `ignore_enum_label_removal` option if the labels are known-dead."
            )
        if orphans:
            logger.info("Ignoring undeclared label(s) %s on enum type %s", orphans, declared_enum.name)

        additions = plan_enum_value_additions(
            declared_enum.name,
            declared_enum.labels,
            defined_labels,
            schema=declared_enum.schema,
        )
        for addition in additions:
            logger.info("Detected AddEnumValueOp for %s.%s", declared_enum.name, addition.value)
            upgrade_ops.ops.append(addition)


def _plan_type_creation(declared_enum: DeclaredEnum, existing_tables: set[str]) -> CreateEnumTypeOp | None:
    """Decide whether an absent enum type needs an explicit ``CREATE TYPE``.

    When every table using the type is also new, ``op.create_table`` creates it as a side effect of emitting the
    column, and adding our own statement would be a duplicate -- fatal, since ``CREATE TYPE`` has no
    ``IF NOT EXISTS``. When some table already exists, nothing creates it: ``op.add_column`` does not fire the
    ``before_create`` event that ``create_table`` does, so without this the migration fails with
    ``type ... does not exist``.

    Table existence is a proxy for "is a ``CreateTableOp`` coming", because the real signal is not available here:
    on Alembic >= 1.18 this comparator runs at ``MEDIUM`` priority *before* the built-in table diff, so
    ``upgrade_ops`` is still empty of ``CreateTableOp``. Reading it would mean registering at ``LAST``, which does
    not exist below 1.18 and would couple this to ``autogen_ordering`` through import order.

    Known limits of the proxy, both erring toward emitting nothing rather than emitting a duplicate: it assumes
    stock table autogeneration is enabled and unfiltered, so a table suppressed by ``include_object`` looks new
    while no ``CreateTableOp`` will appear for it; and it compares unqualified names, so same-named tables in
    different schemas alias.
    """
    tables_that_exist = declared_enum.tables & existing_tables
    if not tables_that_exist:
        return None

    if not declared_enum.tables <= existing_tables:
        raise MixedEnumTypeCreationError(
            f"Enum type {declared_enum.name!r} does not exist yet and is used by both existing table(s) "
            f"{sorted(tables_that_exist)} and new table(s) {sorted(declared_enum.tables - existing_tables)}. "
            "`CREATE TABLE` would create the type implicitly for the new tables while the existing ones need it "
            "created explicitly, and PostgreSQL has no `CREATE TYPE ... IF NOT EXISTS` to reconcile the two. Split "
            "this into separate migrations, or set `create_type=False` on the new tables' ENUM and create it by hand."
        )

    return CreateEnumTypeOp(declared_enum.name, declared_enum.labels, schema=declared_enum.schema)
