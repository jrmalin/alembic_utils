import pytest
from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    desc,
    func,
    literal_column,
    text,
)

from alembic_utils_extended.pg_expression_index import (
    _get_model_indexes,
    _parse_indexdef,
    _truncate_identifier,
)
from alembic_utils_extended.testbase import (
    TEST_VERSIONS_ROOT,
    run_alembic_command,
)


def test_detect_create_expression_index(engine) -> None:
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_name_lower", func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_name_lower"))

    output = run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "create_expr_idx",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_create_expr_idx.py"

    with migration_path.open() as migration_file:
        migration_contents = migration_file.read()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_name_lower" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_drop_expression_index(engine) -> None:
    metadata = MetaData()
    Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("CREATE INDEX idx_test_table_name_lower ON test_table (lower(name))"))

    output = run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "drop_expr_idx",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_drop_expr_idx.py"

    with migration_path.open() as migration_file:
        migration_contents = migration_file.read()

    assert "op.drop_index" in migration_contents
    assert "idx_test_table_name_lower" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_skips_when_expression_index_exists_in_both(engine) -> None:
    """When an expression index exists in both model and DB with the same
    identity (table, name), the comparator treats it as unchanged and emits
    neither a create nor a drop. Content changes must be signaled by rename.
    """
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_name_lower", func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "no_op",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_no_op.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" not in migration_contents
    assert "op.drop_index" not in migration_contents


def test_skips_every_shared_expression_index(engine) -> None:
    """Multiple shared expression indexes are all treated as unchanged; the
    autogen'd migration is empty."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        Column("email", String(100)),
    )
    Index("idx_test_table_name_lower", func.lower(table.c.name))
    Index("idx_test_table_email_lower", func.lower(table.c.email))

    with engine.begin() as connection:
        metadata.create_all(connection)

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "no_op_multi",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_no_op_multi.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" not in migration_contents
    assert "op.drop_index" not in migration_contents


def test_fork_handles_plain_column_indexes(engine) -> None:
    """Fork owns all user-declared indexes, including plain-column ones.
    Model declares idx, DB doesn't have it → fork emits create."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_name", table.c.name)

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_name"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "plain_col_idx",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_plain_col_idx.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_name" in migration_contents


def test_disabled_by_default(engine) -> None:
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_name_lower", func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_name_lower"))

    output = run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "disabled",
        },
        target_metadata=metadata,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_disabled.py"

    with migration_path.open() as migration_file:
        migration_contents = migration_file.read()

    assert "idx_test_table_name_lower" not in migration_contents


def test_get_model_indexes_returns_all_declared_indexes() -> None:
    """Fork iterates every ``Index()`` decl regardless of shape — plain-
    column, function-expression, directional. PK/UNIQUE-backed indexes
    are excluded because they aren't in ``table.indexes``."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_regular", table.c.name)
    Index("idx_expr", func.lower(table.c.name))
    Index("idx_desc", desc(table.c.name))

    indexes = _get_model_indexes(metadata, None)

    assert {idx["name"] for idx in indexes} == {"idx_regular", "idx_expr", "idx_desc"}


def test_truncate_identifier_matches_sqlalchemy_hash_form() -> None:
    """SQLAlchemy hash-truncates any identifier it treats as ``_truncated_label``
    (which includes indexes generated from ``Column(index=True)`` and names
    wrapped in ``op.f(...)``) to fit within PG's 63-char limit. The formula is
    ``name[:55] + "_" + md5(name).hexdigest()[-4:]`` — a 55-char prefix + 4-char
    hash-suffix for the default 63-limit dialect.

    Under 63 chars the name passes through unchanged. Over 63 chars, the
    truncated form must match what SA sends in the CREATE INDEX statement,
    which is what PG stores in ``pg_index`` — otherwise the identity-only
    diff sees model-side long-name vs DB-side hash-truncated as distinct
    and re-emits create ops on every autogen run.
    """
    # Under 63 chars: pass-through.
    short = "ix_short_index_name"
    assert _truncate_identifier(short) == short

    # Exactly 63 chars: pass-through (limit is inclusive).
    at_limit = "a" * 63
    assert _truncate_identifier(at_limit) == at_limit

    # Over 63 chars: hash truncation. Concrete case pulled from candid-api's
    # ``edi999_element_context.component_data_element_position_in_composite``
    # column, whose autogen'd 70-char index name is what SA hash-truncates
    # to ``ix_..._positi_c80b``.
    long_name = "ix_edi999_element_context_component_data_element_position_in_composite"
    expected = "ix_edi999_element_context_component_data_element_positi_c80b"
    truncated = _truncate_identifier(long_name)
    assert truncated == expected
    assert len(truncated) == 60  # 55-char prefix + "_" + 4-char hash.


def test_dev_schema_compared(engine) -> None:
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        schema="DEV",
    )
    Index("idx_test_table_name_lower", func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text('DROP INDEX IF EXISTS "DEV".idx_test_table_name_lower'))

    output = run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "dev_schema",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_dev_schema.py"

    with migration_path.open() as migration_file:
        migration_contents = migration_file.read()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_name_lower" in migration_contents


def test_detect_gin_with_opclass(engine) -> None:
    """GIN index with a trigram opclass baked into the expression. The
    comparator's identity-only diff still detects create/drop by name; verify
    the emitted op preserves ``postgresql_using='gin'`` and the opclass so the
    rebuilt index keeps the trigram search capability."""
    from alembic_utils_extended.pg_extension import PGExtension
    from alembic_utils_extended.replaceable_entity import register_entities

    # Register the pg_trgm extension so PGExtension autogen doesn't emit a
    # drop op for it (the extension is required so the CREATE INDEX using
    # gin_trgm_ops actually applies on upgrade).
    register_entities([PGExtension(schema="public", signature="pg_trgm")], entity_types=[PGExtension])

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index(
        "idx_test_table_name_trgm",
        table.c.name,
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_name_trgm"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "gin_opclass",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_gin_opclass.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_name_trgm" in migration_contents
    assert "postgresql_using" in migration_contents
    assert "gin" in migration_contents
    assert "gin_trgm_ops" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_partial_expression_index(engine) -> None:
    """Partial expression index (postgresql_where clause). Comparator's
    identity-only diff still detects create/drop by name; verify the emitted
    op preserves the WHERE clause so the rebuilt index matches."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index(
        "idx_test_table_lower_name_partial",
        func.lower(table.c.name),
        postgresql_where=text("id > 0"),
    )

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_lower_name_partial"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "partial_expr",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_partial_expr.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_lower_name_partial" in migration_contents
    assert "postgresql_where" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_multi_column_expression_index(engine) -> None:
    """Composite index mixing a plain column with an expression. The
    predicate covers this case (`test_is_expression_index`'s idx_mixed); this
    verifies the end-to-end autogen path emits both elements correctly."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_id_lower_name", table.c.id, func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_id_lower_name"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "multi_col_expr",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_multi_col_expr.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_id_lower_name" in migration_contents
    # Both the column and the lower(name) expression should appear in the op.
    assert "id" in migration_contents
    assert "lower" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_rename_emits_drop_and_create(engine) -> None:
    """An expression-index rename — same expression, different name — should
    surface as drop(old) + create(new). This is the contract that lets you
    express a content-change by renaming, working around the identity-only
    diff's inability to detect expression changes under a stable name."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_new_name", func.lower(table.c.name))

    with engine.begin() as connection:
        metadata.create_all(connection)

    # DB starts with the OLD-name index; model declares NEW-name on the same expression.
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_new_name"))
        connection.execute(text("CREATE INDEX idx_test_table_old_name ON test_table (lower(name))"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "rename_expr_idx",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_rename_expr_idx.py"
    migration_contents = migration_path.read_text()

    assert "op.drop_index" in migration_contents
    assert "idx_test_table_old_name" in migration_contents
    assert "op.create_index" in migration_contents
    assert "idx_test_table_new_name" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_desc_expression_renders_without_table_qualifier(engine) -> None:
    """DESC-modified expressions historically rendered as `tablename.col DESC`,
    which Postgres rejects inside CREATE INDEX. Verify the rendered expression
    strips the table qualifier (Alembic's render_ddl_sql_expr supplies
    include_table=False) and the resulting migration applies cleanly.

    Uses desc() around a real function expression so the fork's comparator
    picks it up — plain desc(Column) is handled by Alembic's stock autogen.
    """
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("idx_test_table_lower_name_desc", desc(func.lower(table.c.name)))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_lower_name_desc"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "desc_expr",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_desc_expr.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_lower_name_desc" in migration_contents
    # The DESC expression must render as `lower(name) DESC`, not `lower(test_table.name) DESC`.
    assert "test_table.name" not in migration_contents
    assert "DESC" in migration_contents

    # And the migration must actually apply — PG rejects table-qualified refs in CREATE INDEX.
    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_raises_on_func_with_string_arg_matching_column_name(engine) -> None:
    """The `func.X("column_name")` anti-pattern (string treated as bound value, not
    column reference) creates an index on a constant string instead of the column.
    Catch it at autogen time rather than letting the wrong index ship silently.

    Pattern in the wild (insurance_card.py): declared at module level alongside one
    real column reference so the Index attaches to the right table, then a sibling
    string-literal Index gets buried with it."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    # The first arg (table.c.id) attaches the Index to the table; the second is the buggy func.
    Index("idx_test_table_name_buggy", table.c.id, func.lower("name"))

    with pytest.raises(ValueError, match=r"anti-pattern"):
        run_alembic_command(
            engine=engine,
            command="revision",
            command_kwargs={
                "autogenerate": True,
                "rev_id": "1",
                "message": "should_raise_anti_pattern",
            },
            target_metadata=metadata,
            compare_indexes=True,
        )


def test_allows_legitimate_string_literal_args(engine) -> None:
    """A string arg that doesn't match a column name is treated as an intentional
    literal — e.g. `func.to_tsvector('english', col)` or `func.replace(col, '-', '')`.
    The anti-pattern check must not false-positive on those."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    # 'english' is not a column name → legitimate literal, should NOT raise.
    # Use postgresql_using='btree' to avoid GIN-on-tsvector's IMMUTABLE requirement.
    Index("idx_test_table_lower_replace", func.replace(func.lower(table.c.name), "-", ""))

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_lower_replace"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "legit_literal",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_legit_literal.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_lower_replace" in migration_contents


def test_detect_postgresql_include_clause_is_preserved(engine) -> None:
    """`postgresql_include=[...]` declares covering columns for index-only scans.
    The comparator must preserve it in the emitted op so the rebuilt index keeps
    the covering optimization."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        Column("payload", String(100)),
    )
    Index(
        "idx_test_table_name_lower_include",
        func.lower(table.c.name),
        postgresql_include=["id", "payload"],
    )

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS idx_test_table_name_lower_include"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={
            "autogenerate": True,
            "rev_id": "1",
            "message": "include_clause",
        },
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_path = TEST_VERSIONS_ROOT / "1_include_clause.py"
    migration_contents = migration_path.read_text()

    assert "op.create_index" in migration_contents
    assert "idx_test_table_name_lower_include" in migration_contents
    assert "postgresql_include" in migration_contents
    # And both INCLUDE columns must round-trip.
    assert "id" in migration_contents
    assert "payload" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Idempotency: autogen must produce a no-op migration on the second run,
# regardless of index shape. This is the safety net for the identity-only
# diff — if it drifts, `cdc alembic` keeps emitting the same ops forever.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index_factory",
    [
        # Function expressions.
        pytest.param(lambda t: Index("ix_t_lower", func.lower(t.c.name)), id="func_lower"),
        pytest.param(lambda t: Index("ix_t_trim_lower", func.trim(func.lower(t.c.name))), id="nested_func"),
        # Direction modifiers.
        pytest.param(lambda t: Index("ix_t_desc", desc(t.c.created_at)), id="desc"),
        pytest.param(lambda t: Index("ix_t_lit_desc", literal_column("created_at DESC")), id="literal_column_desc"),
        pytest.param(lambda t: Index("ix_t_lit_nulls_first", literal_column("created_at NULLS FIRST")), id="nulls_first"),
        # Mixed.
        pytest.param(lambda t: Index("ix_t_mixed", t.c.id, desc(t.c.created_at)), id="mixed_plain_and_desc"),
        pytest.param(lambda t: Index("ix_t_mixed_fn", t.c.id, func.lower(t.c.name)), id="mixed_plain_and_func"),
        # Partial.
        pytest.param(
            lambda t: Index("ix_t_partial", func.lower(t.c.name), postgresql_where=text("id > 0")),
            id="partial_with_expr",
        ),
        # GIN with opclass.
        pytest.param(
            lambda t: Index("ix_t_gin", text("(name || ' ') gin_trgm_ops"), postgresql_using="gin"),
            id="gin_with_opclass",
        ),
    ],
)
def test_autogen_is_idempotent(engine, index_factory) -> None:
    """After applying the fork-generated migration, a second autogen run
    must find no diff (empty upgrade / downgrade blocks). Regression guard
    for the identity-only comparator drifting."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        Column("created_at", String(30)),
    )
    index_factory(table)

    with engine.begin() as connection:
        metadata.create_all(connection)

    # First run: fork emits create ops.
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "first"},
        target_metadata=metadata,
        compare_indexes=True,
    )
    # Apply the generated migration.
    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )

    # Second run: nothing should have changed. Migration body must be empty.
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "2", "message": "second"},
        target_metadata=metadata,
        compare_indexes=True,
    )

    second = (TEST_VERSIONS_ROOT / "2_second.py").read_text()
    assert "op.create_index" not in second, f"Second autogen re-emitted a create:\n{second}"
    assert "op.drop_index" not in second, f"Second autogen re-emitted a drop:\n{second}"
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# NULLS NOT DISTINCT on unique indexes (PostgreSQL 15+). SQLAlchemy 1.4 has no
# native support, so on 1.4 the fork registers the dialect arg and splices the
# clause into CREATE INDEX; on 2.x it defers to native support. The unit tests
# below run on every PG version; the integration tests skip on PG < 15.
# ---------------------------------------------------------------------------


def _skip_if_no_nulls_not_distinct(engine) -> None:
    with engine.connect() as conn:
        if (conn.dialect.server_version_info or (0,)) < (15,):
            pytest.skip("NULLS NOT DISTINCT requires PostgreSQL 15+")


def test_get_model_indexes_extracts_nulls_not_distinct() -> None:
    """A unique index declared with ``postgresql_nulls_not_distinct=True`` carries
    the flag through to the emitted op's kwargs."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("uq_test_table_name", table.c.name, unique=True, postgresql_nulls_not_distinct=True)

    indexes = _get_model_indexes(metadata, None)

    (idx,) = [i for i in indexes if i["name"] == "uq_test_table_name"]
    assert idx["unique"] is True
    assert idx["kw"].get("postgresql_nulls_not_distinct") is True


def test_nulls_not_distinct_on_non_unique_raises() -> None:
    """PostgreSQL accepts NULLS NOT DISTINCT on any index, but it only affects
    uniqueness — on a non-unique index it's a silent no-op, almost always a
    forgotten unique=True. The fork fails fast at autogen time rather than shipping
    an index that doesn't do what the declaration implies."""
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("ix_test_table_name", table.c.name, postgresql_nulls_not_distinct=True)

    with pytest.raises(ValueError, match=r"no effect on a non-unique index"):
        _get_model_indexes(metadata, None)


def test_parse_indexdef_detects_nulls_not_distinct() -> None:
    """``_parse_indexdef`` reads NULLS NOT DISTINCT out of ``pg_get_indexdef``
    output so a drop's reverse (downgrade) op recreates the index with the flag."""
    with_nnd = "CREATE UNIQUE INDEX uq_t ON public.t USING btree (name) NULLS NOT DISTINCT"
    _, kw = _parse_indexdef(with_nnd)
    assert kw.get("postgresql_nulls_not_distinct") is True

    without_nnd = "CREATE UNIQUE INDEX uq_t ON public.t USING btree (name)"
    _, kw_plain = _parse_indexdef(without_nnd)
    assert "postgresql_nulls_not_distinct" not in kw_plain


def test_detect_create_unique_index_nulls_not_distinct(engine) -> None:
    """Model declares a NULLS NOT DISTINCT unique index the DB lacks → the fork
    emits a create op carrying the flag, and the applied index actually carries
    NULLS NOT DISTINCT in the live catalog."""
    _skip_if_no_nulls_not_distinct(engine)
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("uq_test_table_name_nnd", table.c.name, unique=True, postgresql_nulls_not_distinct=True)

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS uq_test_table_name_nnd"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "create_nnd"},
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_create_nnd.py").read_text()
    assert "op.create_index" in migration_contents
    assert "uq_test_table_name_nnd" in migration_contents
    assert "unique=True" in migration_contents
    assert "postgresql_nulls_not_distinct=True" in migration_contents

    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )

    with engine.begin() as connection:
        indexdef = connection.execute(
            text(
                "SELECT pg_get_indexdef(i.indexrelid) FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = 'uq_test_table_name_nnd'"
            )
        ).scalar()
    assert indexdef is not None and "NULLS NOT DISTINCT" in indexdef

    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )


def test_detect_drop_and_downgrade_recreates_nulls_not_distinct(engine) -> None:
    """DB has a NULLS NOT DISTINCT unique index the model doesn't declare → drop.
    The reverse op must recreate it with the flag intact on downgrade."""
    _skip_if_no_nulls_not_distinct(engine)
    metadata = MetaData()
    Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )

    with engine.begin() as connection:
        metadata.create_all(connection)

    with engine.begin() as connection:
        connection.execute(text("CREATE UNIQUE INDEX uq_test_table_name_nnd ON test_table (name) NULLS NOT DISTINCT"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "drop_nnd"},
        target_metadata=metadata,
        compare_indexes=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_drop_nnd.py").read_text()
    assert "op.drop_index" in migration_contents
    assert "uq_test_table_name_nnd" in migration_contents
    assert "postgresql_nulls_not_distinct=True" in migration_contents

    # upgrade drops it...
    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
        target_metadata=metadata,
    )
    with engine.begin() as connection:
        exists = connection.execute(text("SELECT 1 FROM pg_class WHERE relname = 'uq_test_table_name_nnd'")).scalar()
    assert exists is None

    # ...downgrade recreates it with NULLS NOT DISTINCT.
    run_alembic_command(
        engine=engine,
        command="downgrade",
        command_kwargs={"revision": "base"},
        target_metadata=metadata,
    )
    with engine.begin() as connection:
        indexdef = connection.execute(
            text(
                "SELECT pg_get_indexdef(i.indexrelid) FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = 'uq_test_table_name_nnd'"
            )
        ).scalar()
    assert indexdef is not None and "NULLS NOT DISTINCT" in indexdef


def test_nulls_not_distinct_autogen_is_idempotent(engine) -> None:
    """With the NULLS NOT DISTINCT unique index present in both model and DB, autogen
    must be a no-op. Exercises the create_all render path (the fork's shim on 1.4)
    plus the identity-only diff staying stable."""
    _skip_if_no_nulls_not_distinct(engine)
    metadata = MetaData()
    table = Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Index("uq_test_table_name_nnd", table.c.name, unique=True, postgresql_nulls_not_distinct=True)

    with engine.begin() as connection:
        metadata.create_all(connection)

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "idem_nnd"},
        target_metadata=metadata,
        compare_indexes=True,
    )

    contents = (TEST_VERSIONS_ROOT / "1_idem_nnd.py").read_text()
    assert "op.create_index" not in contents, f"Autogen re-emitted a create:\n{contents}"
    assert "op.drop_index" not in contents, f"Autogen re-emitted a drop:\n{contents}"


def test_mv_index_not_duplicated_by_compare_indexes(engine) -> None:
    """compare_indexes must not emit a separate CreateIndexOp for an index that belongs
    to a registered PGMaterializedView. render_post_create_entity already emits
    op.create_index inline — a second op from compare_indexes would make the migration
    fail (index already exists) or create a redundant migration file."""
    from alembic_utils_extended.pg_materialized_view import PGMaterializedView
    from alembic_utils_extended.replaceable_entity import register_entities

    metadata = MetaData()
    table = Table(
        "test_mv",
        metadata,
        Column("id", Integer, primary_key=True),
    )
    idx = Index("ix_test_mv_id", table.c.id)

    mv = PGMaterializedView(
        schema="public",
        signature="test_mv",
        definition="SELECT 1 AS id FROM generate_series(1, 1)",
        with_data=True,
        indexes=[idx],
    )
    register_entities([mv], entity_types=[PGMaterializedView])

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "mv_idx_dedup"},
        target_metadata=metadata,
        compare_indexes=True,
    )

    contents = (TEST_VERSIONS_ROOT / "1_mv_idx_dedup.py").read_text()

    # render_post_create_entity produces exactly one op.create_index.
    # compare_indexes must not add a second one for the same index.
    assert contents.count("op.create_index") == 1, f"Expected 1 op.create_index (from entity create inline), found duplicates:\n{contents}"
