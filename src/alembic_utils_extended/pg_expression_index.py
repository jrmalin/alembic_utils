from __future__ import annotations

import hashlib
import logging
import re
from typing import TypedDict

import sqlalchemy
from alembic.autogenerate import comparators
from alembic.autogenerate.api import AutogenContext
from alembic.operations import ops
from sqlalchemy import Column, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateIndex, Index
from sqlalchemy.sql import visitors
from sqlalchemy.sql.elements import BindParameter, TextClause

from alembic_utils_extended.pg_materialized_view import PGMaterializedView
from alembic_utils_extended.replaceable_entity import registry

logger = logging.getLogger(__name__)

# ``NULLS [NOT] DISTINCT`` on a unique index (PostgreSQL 15+) is a native
# ``postgresql_nulls_not_distinct`` dialect option from SQLAlchemy 2.0 onward.
# SQLAlchemy 1.4 has no support at all: ``Index(...)`` rejects the kwarg and its
# PG ``visit_create_index`` compiler never emits the clause. We run PG 17 but our
# production SQLAlchemy is 1.4, so on 1.4 we replicate 2.0's behavior — register
# the dialect argument so the kwarg constructs, and splice the clause into the
# compiled ``CREATE INDEX`` ourselves. On 2.x we defer entirely to native support.
_SA_MAJOR = int(sqlalchemy.__version__.split(".")[0])

if _SA_MAJOR < 2:
    Index.argument_for("postgresql", "nulls_not_distinct", None)

    @compiles(CreateIndex, "postgresql")
    def _compile_create_index_nulls_not_distinct(create, compiler, **kw):  # pylint: disable=unused-variable
        # Call the compiler's bound method directly (not the ``@compiles``
        # dispatch), so this is the original PG rendering with no recursion.
        ddl = compiler.visit_create_index(create, **kw)
        nnd = create.element.dialect_options["postgresql"]["nulls_not_distinct"]
        if nnd is None or "NULLS NOT DISTINCT" in ddl or "NULLS DISTINCT" in ddl:
            return ddl
        clause = " NULLS NOT DISTINCT" if nnd else " NULLS DISTINCT"
        # PG grammar places the clause after INCLUDE(...) and before
        # WITH / TABLESPACE / WHERE. Insert ahead of the first of those, if any.
        for marker in (" WITH (", " TABLESPACE ", " WHERE "):
            at = ddl.find(marker)
            if at != -1:
                return ddl[:at] + clause + ddl[at:]
        return ddl + clause


# PostgreSQL's default NAMEDATALEN is 64, giving a usable identifier limit of
# 63 characters. SQLAlchemy applies a hash-based truncation to any identifier
# it treats as a ``_truncated_label`` (which includes indexes generated from
# ``Column(index=True)`` and names wrapped in ``op.f(...)``): the CREATE INDEX
# statement it sends to PG uses ``name[:max_-8] + "_" + md5(name)[-4:]`` — a
# 55-char prefix + 4-char hash for the default 63-limit dialect. PG stores
# whatever SA sent verbatim, so ``pg_index.relname`` for these indexes is the
# hashed form. Model-side names come from Python and are the full untruncated
# string; normalizing both sides through this function gives a stable key for
# identity-only diff.
#
# See ``IdentifierPreparer._truncate_and_render_maxlen_name`` in
# ``sqlalchemy/sql/compiler.py`` for the source.
_PG_MAX_IDENTIFIER_LENGTH = 63


def _truncate_identifier(name: str) -> str:
    if len(name) <= _PG_MAX_IDENTIFIER_LENGTH:
        return name
    return name[: _PG_MAX_IDENTIFIER_LENGTH - 8] + "_" + hashlib.md5(name.encode()).hexdigest()[-4:]


class _IndexKw(TypedDict, total=False):
    postgresql_using: str
    postgresql_ops: dict[str, str]
    postgresql_where: str
    postgresql_include: list[str]
    postgresql_nulls_not_distinct: bool


class _IndexInfo(TypedDict):
    table_name: str
    name: str
    columns: list[str | TextClause]
    expressions: list[str]
    unique: bool
    kw: _IndexKw


@comparators.dispatch_for("schema")
def compare_indexes(
    autogen_context: AutogenContext,
    upgrade_ops: ops.UpgradeOps,
    _schemas: list[str | None],
) -> None:
    """Autogen index diff on identity ``(table_name, index_name)`` only.

    Owns autogen for ALL user-declared indexes when ``compare_indexes=True``
    is set. Stock Alembic's index dispatcher is buggy or absent on
    SQLAlchemy 1.4 across many shapes — function expressions, directional
    modifiers, ``postgresql_ops`` hacks, opclasses — so this comparator
    takes the whole namespace and diffs by name.

    Consumers should register an ``include_object`` filter in their
    ``env.py`` returning ``False`` for ``type_ == "index"`` to prevent
    stock Alembic from dueling on the same indexes. Indexes backing PK /
    UNIQUE constraints are excluded automatically (they're managed by
    stock Alembic's constraint diff, not the index diff).

    Optional ``compare_indexes_include(table_name, index_name, reflected)``
    callback can be set in ``config.attributes`` to exclude specific
    indexes from the fork's scope (e.g., sqlalchemy-continuum's ``_version``
    tables whose indexes are managed by continuum, not the app schema).

    Content changes are not detected. To evolve an index's columns,
    WHERE clause, INCLUDE list, opclass, or method, rename it — that
    produces a drop + create pair the comparator will emit.
    """
    if not autogen_context.opts.get("compare_indexes"):
        return

    inspector: Inspector = autogen_context.inspector
    target_metadata = autogen_context.metadata

    if target_metadata is None:
        return

    include_index = autogen_context.opts.get("compare_indexes_include") or (lambda *args, **kw: True)

    observed_schemas: set[str | None] = {table.schema for table in target_metadata.tables.values()}
    mv_table_names = {entity.signature for entity in registry.entities() if isinstance(entity, PGMaterializedView)}

    for schema_to_use in observed_schemas:

        model_indexes = [
            idx
            for idx in _get_model_indexes(target_metadata, schema_to_use, autogen_context)
            if include_index(idx["table_name"], idx["name"], False)
        ]
        db_indexes = [
            idx
            for idx in _get_database_indexes(inspector, target_metadata, schema_to_use)
            if include_index(idx["table_name"], idx["name"], True)
        ]

        # Match on the PG-truncated name (identifiers over ``NAMEDATALEN - 1 =
        # 63`` are silently truncated at CREATE time). Model-side names come
        # from Python and may exceed the limit; DB-side names are already
        # truncated by PG. Normalize both to the same key so identity match
        # actually matches.
        def _key(idx):
            return (idx["table_name"], _truncate_identifier(idx["name"]))

        model_keys = {_key(idx) for idx in model_indexes}
        db_keys = {_key(idx) for idx in db_indexes}

        create_keys = model_keys - db_keys
        drop_keys = db_keys - model_keys

        create_keys = {key for key in create_keys if key[0] not in mv_table_names}

        # Indexes present in both are assumed unchanged (identity-only diff).

        for key in create_keys:
            index_info = next(idx for idx in model_indexes if _key(idx) == key)
            table_name = index_info["table_name"]
            index_name = index_info["name"]
            logger.info(
                "Detected CreateIndexOp for %s.%s",
                table_name,
                index_name,
            )
            create_op = ops.CreateIndexOp(
                index_name=index_name,
                table_name=table_name,
                columns=index_info["columns"],
                schema=schema_to_use,
                unique=index_info.get("unique", False),
                **index_info.get("kw", {}),
            )
            upgrade_ops.ops.append(create_op)

        for key in drop_keys:
            index_info = next(idx for idx in db_indexes if _key(idx) == key)
            table_name = index_info["table_name"]
            index_name = index_info["name"]
            logger.info(
                "Detected DropIndexOp for %s.%s",
                table_name,
                index_name,
            )
            create_op_for_reverse = ops.CreateIndexOp(
                index_name=index_name,
                table_name=table_name,
                columns=index_info["columns"],
                schema=schema_to_use,
                unique=index_info.get("unique", False),
                **index_info.get("kw", {}),
            )
            # ``DropIndexOp.to_index()`` (used to render the downgrade's reverse
            # CreateIndex) takes columns from ``_reverse`` but reads ``unique``
            # and the dialect kwargs from the DropIndexOp's OWN ``kw`` — so they
            # must be passed here, not only on ``create_op_for_reverse``, or the
            # downgrade recreates the index without unique / using / where /
            # include / nulls_not_distinct.
            drop_op = ops.DropIndexOp(
                index_name=index_name,
                table_name=table_name,
                schema=schema_to_use,
                _reverse=create_op_for_reverse,
                unique=index_info.get("unique", False),
                **index_info.get("kw", {}),
            )
            upgrade_ops.ops.append(drop_op)


def _get_model_indexes(metadata, schema: str | None, autogen_context: AutogenContext | None = None) -> list[_IndexInfo]:
    """Extract every user-declared :class:`Index` from ``metadata`` for the
    given schema.

    ``table.indexes`` on a SQLAlchemy :class:`Table` only contains indexes
    declared via ``Index(...)``; the implicit indexes backing PRIMARY KEY
    and UNIQUE constraints are managed separately by stock Alembic's
    constraint diff, so this iteration naturally excludes them.
    """
    indexes: list[_IndexInfo] = []

    for table in metadata.tables.values():
        if table.schema != schema:
            continue

        for index in table.indexes:
            if index.name is None:
                raise ValueError(f"Unnamed index on table '{table.name}'. " f"All indexes must have a name for autogenerate support.")

            columns = []
            expressions_list = []
            for expr in index.expressions:
                if isinstance(expr, Column):
                    columns.append(expr.name)
                    expressions_list.append(expr.name)
                else:
                    _raise_if_string_arg_matches_column_name(expr, table, index.name)
                    expr_str = _render_index_expression(expr, autogen_context)
                    columns.append(text(expr_str))
                    expressions_list.append(expr_str)

            kw: _IndexKw = {}
            if hasattr(index, "dialect_options") and "postgresql" in index.dialect_options:
                pg_opts = index.dialect_options["postgresql"]
                if pg_opts.get("using"):
                    kw["postgresql_using"] = pg_opts["using"]
                if pg_opts.get("ops"):
                    kw["postgresql_ops"] = pg_opts["ops"]
                if pg_opts.get("where") is not None:
                    kw["postgresql_where"] = _render_index_expression(pg_opts["where"], autogen_context)
                if pg_opts.get("include"):
                    kw["postgresql_include"] = list(pg_opts["include"])
                if pg_opts.get("nulls_not_distinct"):
                    # PostgreSQL accepts NULLS NOT DISTINCT on any index, but it only
                    # affects uniqueness — on a non-unique index it is a silent no-op,
                    # almost always a forgotten unique=True. Fail fast rather than ship
                    # an index that doesn't do what the declaration implies.
                    if not index.unique:
                        raise ValueError(
                            f"Index {index.name!r} on table {table.name!r} sets nulls_not_distinct "
                            "but is not unique. NULLS NOT DISTINCT only affects uniqueness, so it has "
                            "no effect on a non-unique index — did you mean to set unique=True?"
                        )
                    kw["postgresql_nulls_not_distinct"] = True

            indexes.append(
                {
                    "table_name": table.name,
                    "name": index.name,
                    "columns": columns,
                    "expressions": expressions_list,
                    "unique": index.unique,
                    "kw": kw,
                }
            )

    return indexes


def _raise_if_string_arg_matches_column_name(expr, table, index_name: str) -> None:
    """Catch the ``func.X("column_name_str")`` anti-pattern.

    SQLAlchemy treats bare strings inside ``func.X(...)`` as bound-parameter LITERAL
    VALUES, not column references — ``func.lower("payer_name")`` compiles to
    ``lower(:literal_1)`` with the value ``'payer_name'``, NOT to ``lower(payer_name)``.
    The resulting index is on the constant string and useless. Catch it at autogen
    time by walking the expression tree for BindParameter values whose strings
    match a column name on the index's table — that combination is almost always
    the bug rather than an intentional literal.
    """
    column_names = {col.name for col in table.columns}
    for elem in visitors.iterate(expr):
        if isinstance(elem, BindParameter) and isinstance(elem.value, str) and elem.value in column_names:
            raise ValueError(
                f"Index {index_name!r} on table {table.name!r} references a string "
                f"literal {elem.value!r} that matches a column on the same table. This is the "
                f"`func.X({elem.value!r})` anti-pattern — the string is being bound as a parameter "
                f"value, not a column reference, so the index is on the constant string. Use "
                f"`func.X(table.c.{elem.value})` or `func.X(literal_column({elem.value!r}))` instead.",
            )


def _render_index_expression(expr, autogen_context: AutogenContext | None) -> str:
    """Render a SQLAlchemy expression as a SQL string suitable for CREATE INDEX.

    Delegates to Alembic's ``render_ddl_sql_expr``, which supplies the right compile
    flags (``literal_binds=True``, ``include_table=False``) and the postgres-dialect-aware
    handling of index expression self-grouping. Falls back to a plain ``str(expr)`` only
    when no autogen_context is supplied (test paths exercising the predicate alone)."""
    if autogen_context is None or not hasattr(expr, "compile"):
        return str(expr)
    return autogen_context.migration_context.impl.render_ddl_sql_expr(expr, is_index=True)


def _parse_indexdef(indexdef: str) -> tuple[list[TextClause], _IndexKw]:
    """Parse a ``pg_get_indexdef`` output into ``(columns, kwargs)`` suitable
    for reconstructing a :class:`CreateIndexOp` on downgrade.

    ``pg_get_indexdef`` shape:
        ``CREATE [UNIQUE] INDEX name ON [schema.]table USING method (cols) [INCLUDE (...)] [WHERE ...]``

    Column-list splitting is depth-aware so expressions with commas (e.g.
    ``coalesce(a, b)``) survive intact.
    """
    kwargs: _IndexKw = {}

    method_match = re.search(r"USING\s+(\w+)\s+\(", indexdef, re.IGNORECASE)
    if method_match is None:
        return ([text(indexdef)], kwargs)
    kwargs["postgresql_using"] = method_match.group(1)

    # Walk from just past the opening paren to find its matching close.
    start = method_match.end()
    depth = 1
    i = start
    while i < len(indexdef) and depth > 0:
        char = indexdef[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        i += 1
    col_list_str = indexdef[start : i - 1]
    tail = indexdef[i:]

    # Split the column list on top-level commas only.
    columns: list[TextClause] = []
    buf = ""
    depth = 0
    for char in col_list_str:
        if char == "(":
            depth += 1
            buf += char
        elif char == ")":
            depth -= 1
            buf += char
        elif char == "," and depth == 0:
            columns.append(text(buf.strip()))
            buf = ""
        else:
            buf += char
    if buf.strip():
        columns.append(text(buf.strip()))

    include_match = re.search(r"INCLUDE\s+\(([^)]+)\)", tail, re.IGNORECASE)
    if include_match:
        kwargs["postgresql_include"] = [c.strip() for c in include_match.group(1).split(",")]

    # ``pg_get_indexdef`` emits ``NULLS NOT DISTINCT`` between INCLUDE and WHERE on
    # PG 15+; PG < 15 never emits it, so this is safe without a server-version check.
    if re.search(r"\bNULLS\s+NOT\s+DISTINCT\b", tail, re.IGNORECASE):
        kwargs["postgresql_nulls_not_distinct"] = True

    where_match = re.search(r"WHERE\s+(.+)$", tail.strip(), re.IGNORECASE | re.DOTALL)
    if where_match:
        kwargs["postgresql_where"] = where_match.group(1).strip()

    return (columns, kwargs)


def _get_database_indexes(
    inspector: Inspector,
    metadata,
    schema: str | None,
) -> list[_IndexInfo]:
    """Read every user-declared index in the target schema directly from
    ``pg_index``.

    Bypasses :meth:`Inspector.get_indexes` because SQLAlchemy 1.4's
    reflector silently skips function-expression indexes on PostgreSQL
    (emits ``SAWarning: Skipped unsupported reflection ...``). Querying
    ``pg_index`` directly captures everything uniformly.

    Excludes indexes backing PRIMARY KEY and UNIQUE constraints
    (identified via ``pg_constraint.conindid``) — those are managed by
    stock Alembic's constraint diff, not the index diff.
    """
    table_names = [t.name for t in metadata.tables.values() if t.schema == schema]
    if not table_names:
        return []

    query = text(
        """
        SELECT
            c.relname AS index_name,
            t.relname AS table_name,
            pg_get_indexdef(i.indexrelid) AS index_definition,
            i.indisunique AS is_unique
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_constraint con ON con.conindid = i.indexrelid
        WHERE n.nspname = :schema
          AND t.relname = ANY(:table_names)
          AND con.oid IS NULL
          AND NOT i.indisprimary
        """
    )

    resolved_schema = schema or "public"
    rows = inspector.bind.execute(query, {"schema": resolved_schema, "table_names": table_names}).fetchall()

    indexes: list[_IndexInfo] = []
    for row in rows:
        columns, kw = _parse_indexdef(row.index_definition)
        indexes.append(
            {
                "table_name": row.table_name,
                "name": row.index_name,
                "columns": columns,
                "expressions": [str(col) for col in columns],
                "unique": row.is_unique,
                "kw": kw,
            }
        )
    return indexes
