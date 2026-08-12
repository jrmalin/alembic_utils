"""Migration operations for native PostgreSQL enum types.

Kept separate from ``pg_enum`` so that ``autogen_ordering`` can import the op classes for its ``isinstance`` buckets
without a circular import back through the comparator.

Every op here subclasses :class:`alembic.operations.ops.ExecuteSQLOp`, which buys rendering for free: alembic's
renderer dispatch walks ``type(obj).__mro__``, so each one renders as ``op.execute('...')`` with no registration.

Every op here also implements ``reverse()``, and that is load-bearing rather than polish.
``alembic.autogenerate.compare._populate_migration_script`` calls ``upgrade_ops.reverse_into(downgrade_ops)``
unconditionally after the comparators run, and ``ExecuteSQLOp`` inherits ``MigrateOperation.reverse()``, which raises
``NotImplementedError``. An op without a working ``reverse()`` therefore crashes ``alembic revision --autogenerate``
and ``alembic check`` outright, whether or not anyone wants a downgrade.
"""
from __future__ import annotations

from alembic.autogenerate import renderers
from alembic.autogenerate.api import AutogenContext
from alembic.autogenerate.render import _alembic_autogenerate_prefix
from alembic.operations import ops
from sqlalchemy.dialects import postgresql

_PREPARER = postgresql.dialect().identifier_preparer


def quote_literal(value: str) -> str:
    """Render a string as a single-quoted SQL literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def quote_identifier(name: str, schema: str | None = None) -> str:
    """Render a possibly schema-qualified identifier, quoting only where postgres requires it.

    Leaving ordinary lowercase names bare keeps generated statements identical to the hand-written ones they replace,
    and keeps the rendered python string free of double quotes so it can use the nicer literal form below.
    """
    quoted = _PREPARER.quote(name)
    return quoted if schema is None else f"{_PREPARER.quote_schema(schema)}.{quoted}"


def _render_execute(autogen_context: AutogenContext, op: CreateEnumTypeOp | DropEnumTypeOp | AddEnumValueOp) -> str:
    """Render ``op.execute(...)`` preferring a double-quoted python literal.

    Alembic's stock renderer uses ``{sqltext!r}``, which picks single quotes and then backslash-escapes every SQL
    string literal inside -- ``op.execute('ALTER TYPE letters ADD VALUE \\'b\\' BEFORE \\'c\\'')``. Since these
    statements are full of quoted labels, and the migrations land in front of human reviewers, emit the readable
    form instead whenever the SQL contains no double quote of its own.
    """
    statement = str(op.sqltext)
    literal = f'"{statement}"' if '"' not in statement else repr(statement)
    return f"{_alembic_autogenerate_prefix(autogen_context)}execute({literal})"


class CreateEnumTypeOp(ops.ExecuteSQLOp):
    """``CREATE TYPE <name> AS ENUM (...)``.

    PostgreSQL has no ``CREATE TYPE ... IF NOT EXISTS``; a duplicate raises SQLSTATE 42710. The comparator is
    therefore responsible for only emitting this when the type is genuinely absent and nothing else is about to
    create it implicitly.
    """

    def __init__(self, type_name: str, labels: tuple[str, ...], *, schema: str | None = None) -> None:
        self.type_name = type_name
        self.labels = tuple(labels)
        self.schema = schema
        rendered_labels = ", ".join(quote_literal(label) for label in self.labels)
        super().__init__(f"CREATE TYPE {quote_identifier(type_name, schema)} AS ENUM ({rendered_labels})")

    def reverse(self) -> DropEnumTypeOp:
        return DropEnumTypeOp(self)


class DropEnumTypeOp(ops.ExecuteSQLOp):
    """``DROP TYPE <name>``. Only ever produced as the reverse of :class:`CreateEnumTypeOp`.

    Holds the op it undoes rather than copying its fields, so reversing back returns the original.
    """

    def __init__(self, create_op: CreateEnumTypeOp) -> None:
        self.create_op = create_op
        super().__init__(f"DROP TYPE {quote_identifier(create_op.type_name, create_op.schema)}")

    def reverse(self) -> CreateEnumTypeOp:
        return self.create_op


class AddEnumValueOp(ops.ExecuteSQLOp):
    """``ALTER TYPE <name> ADD VALUE IF NOT EXISTS <value> [BEFORE|AFTER <neighbor>]``.

    Legal inside a transaction block since PostgreSQL 12; only *using* the new value before the transaction commits
    is barred, which a migration that merely adds it never does.
    """

    def __init__(
        self,
        type_name: str,
        value: str,
        *,
        before: str | None = None,
        after: str | None = None,
        schema: str | None = None,
    ) -> None:
        if before is not None and after is not None:
            raise ValueError(f"AddEnumValueOp for {type_name}.{value} cannot set both `before` and `after`.")

        self.type_name = type_name
        self.value = value
        self.before = before
        self.after = after
        self.schema = schema

        statement = f"ALTER TYPE {quote_identifier(type_name, schema)} ADD VALUE IF NOT EXISTS {quote_literal(value)}"
        if before is not None:
            statement += f" BEFORE {quote_literal(before)}"
        elif after is not None:
            statement += f" AFTER {quote_literal(after)}"
        super().__init__(statement)

    def reverse(self) -> UnreversibleEnumValueOp:
        return UnreversibleEnumValueOp(self)


class UnreversibleEnumValueOp(ops.MigrateOperation):
    """Placeholder standing in for the downgrade of an :class:`AddEnumValueOp`.

    PostgreSQL cannot remove a label from an enum type -- undoing one requires recreating the type and rewriting
    every dependent table. Rather than emit something destructive, or raise and take autogenerate down with it, this
    renders to nothing and lets alembic write ``pass``.
    """

    def __init__(self, add_op: AddEnumValueOp) -> None:
        self.add_op = add_op

    def reverse(self) -> AddEnumValueOp:
        return self.add_op


@renderers.dispatch_for(UnreversibleEnumValueOp)
def _render_unreversible_enum_value(_autogen_context: AutogenContext, _op: UnreversibleEnumValueOp) -> list[str]:
    return []


# Registered per concrete class so the inherited `ExecuteSQLOp` renderer still serves every other raw-SQL op; renderer
# dispatch walks the MRO. These keys are unclaimed, so the decorator works -- unlike `pg_check_constraint`, which has
# to write `renderers._registry` directly because alembic pre-registers `CreateCheckConstraintOp`.
renderers.dispatch_for(CreateEnumTypeOp)(_render_execute)
renderers.dispatch_for(DropEnumTypeOp)(_render_execute)
renderers.dispatch_for(AddEnumValueOp)(_render_execute)
