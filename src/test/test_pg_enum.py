import enum

import pytest
from sqlalchemy import Column, Enum, Integer, MetaData, Table, text

from alembic_utils_extended.pg_enum import (
    AmbiguousEnumDeclarationError,
    EnumLabelRemovedError,
    MixedEnumTypeCreationError,
    collect_declared_native_enums,
    plan_enum_value_additions,
)
from alembic_utils_extended.pg_enum_ops import AddEnumValueOp, CreateEnumTypeOp
from alembic_utils_extended.testbase import (
    TEST_VERSIONS_ROOT,
    run_alembic_command,
)


def apply_additions(defined: tuple[str, ...], additions: list[AddEnumValueOp]) -> list[str]:
    """Model how postgres mutates the label order, so tests assert on the outcome rather than the SQL."""
    labels = list(defined)
    for addition in additions:
        if addition.before is not None:
            labels.insert(labels.index(addition.before), addition.value)
        elif addition.after is not None:
            labels.insert(labels.index(addition.after) + 1, addition.value)
        else:
            labels.append(addition.value)
    return labels


def order_is_consistent(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    shared = set(left) & set(right)
    return [label for label in left if label in shared] == [label for label in right if label in shared]


# (case, declared, defined). Named cases reference the candid-api migration they were taken from.
ADDITION_CASES = [
    # Two labels inserted before the same anchor. Successive BEFORE stacks in emission order, so declared order holds.
    (
        "multi_before",
        ("created", "eligible_for_post", "queued", "posting_started", "posting_failed"),
        ("created", "posting_started", "posting_failed"),
    ),
    ("single_append", ("WC", "ZZ"), ("WC",)),
    # A run of new labels at the tail. Anchored with AFTER, which reverses unless each addition re-anchors on the
    # previous one -- the regression guard for the running-state part of the algorithm.
    ("tail_run", ("X", "p", "q", "r"), ("X",)),
    ("leading_run", ("a", "b", "Z"), ("Z",)),
    ("interleaved_run", ("a", "1", "b", "2", "c"), ("a", "b", "c")),
    # The database's overall order has drifted, but the new label still lands beside its declared neighbour.
    ("drifted_database", ("DE", "FE", "FC", "FL", "GA"), ("GA", "DE", "FE", "FL")),
    ("fully_inverted", ("A", "B", "C"), ("C", "A")),
    ("no_change", ("a", "b"), ("a", "b")),
    ("empty_type", ("a", "b"), ()),
]


@pytest.mark.parametrize(("case", "declared", "defined"), ADDITION_CASES, ids=[case[0] for case in ADDITION_CASES])
def test_plan_adds_every_missing_label(case: str, declared: tuple[str, ...], defined: tuple[str, ...]) -> None:
    result = apply_additions(defined, plan_enum_value_additions("t", declared, defined))

    assert set(declared) <= set(result), f"{case}: not every declared label was added"
    assert len(result) == len(set(result)), f"{case}: a label was added twice"


@pytest.mark.parametrize(("case", "declared", "defined"), ADDITION_CASES, ids=[case[0] for case in ADDITION_CASES])
def test_plan_places_new_labels_in_declared_order(case: str, declared: tuple[str, ...], defined: tuple[str, ...]) -> None:
    """Newly added labels must end up ordered against each other exactly as declared."""
    additions = plan_enum_value_additions("t", declared, defined)
    added = {addition.value for addition in additions}
    result = apply_additions(defined, additions)

    assert [label for label in result if label in added] == [label for label in declared if label in added], case


@pytest.mark.parametrize(("case", "declared", "defined"), ADDITION_CASES, ids=[case[0] for case in ADDITION_CASES])
def test_plan_reproduces_declared_order_when_database_has_not_drifted(
    case: str, declared: tuple[str, ...], defined: tuple[str, ...]
) -> None:
    """When the existing order already agrees with the models, the result must match the declaration exactly.

    A drifted database is explicitly out of scope: ADD VALUE cannot move an existing label, so those cases only
    promise the weaker guarantee asserted above.
    """
    if not order_is_consistent(declared, defined):
        pytest.skip(f"{case}: database order already drifted, global order is not recoverable")

    result = apply_additions(defined, plan_enum_value_additions("t", declared, defined))

    assert result == list(declared), case


def test_plan_reproduces_a_known_hand_written_migration() -> None:
    """The `multi_before` case, asserted as the literal SQL a human wrote for it."""
    additions = plan_enum_value_additions(
        "era_payment_post_status_enum",
        ("created", "eligible_for_post", "queued", "posting_started"),
        ("created", "posting_started"),
    )

    assert [addition.sqltext for addition in additions] == [
        "ALTER TYPE era_payment_post_status_enum ADD VALUE IF NOT EXISTS 'eligible_for_post' BEFORE 'posting_started'",
        "ALTER TYPE era_payment_post_status_enum ADD VALUE IF NOT EXISTS 'queued' BEFORE 'posting_started'",
    ]


def test_plan_returns_nothing_when_in_sync() -> None:
    assert plan_enum_value_additions("t", ("a", "b"), ("a", "b")) == []


def test_plan_escapes_quotes_in_labels_and_anchors() -> None:
    (addition,) = plan_enum_value_additions("it's", ("o'clock", "z"), ("z",))

    assert addition.sqltext == """ALTER TYPE "it's" ADD VALUE IF NOT EXISTS 'o''clock' BEFORE 'z'"""


def test_collect_finds_enums_that_never_say_native_enum() -> None:
    """`native_enum` defaults to True, so omitting it still declares a native type."""
    metadata = MetaData()
    Table("t", metadata, Column("c", Enum("a", "b", name="implicitly_native")))

    declared = collect_declared_native_enums(metadata)

    assert declared[(None, "implicitly_native")].labels == ("a", "b")


def test_collect_ignores_non_native_enums() -> None:
    metadata = MetaData()
    Table("t", metadata, Column("c", Enum("a", "b", name="check_backed", native_enum=False)))

    assert collect_declared_native_enums(metadata) == {}


def test_collect_resolves_values_callable() -> None:
    """Labels come from `.enums`, which is what actually reaches postgres."""

    class Colour(enum.Enum):
        RED = "red"
        BLUE = "blue"

    by_name = MetaData()
    Table("t", by_name, Column("c", Enum(Colour, name="colour")))
    by_value = MetaData()
    Table("t", by_value, Column("c", Enum(Colour, name="colour", values_callable=lambda e: [m.value for m in e])))

    assert collect_declared_native_enums(by_name)[(None, "colour")].labels == ("RED", "BLUE")
    assert collect_declared_native_enums(by_value)[(None, "colour")].labels == ("red", "blue")


def test_collect_deduplicates_repeated_declarations() -> None:
    """Copied columns -- version/shadow tables -- declare independent Enum objects for one postgres type."""
    metadata = MetaData()
    Table("t", metadata, Column("c", Enum("a", "b", name="shared")))
    Table("t_version", metadata, Column("c", Enum("a", "b", name="shared")))

    declared = collect_declared_native_enums(metadata)

    assert list(declared) == [(None, "shared")]
    assert declared[(None, "shared")].tables == {"t", "t_version"}
    assert declared[(None, "shared")].declared_by == [("t", "c"), ("t_version", "c")]
    assert declared[(None, "shared")].describe_columns() == "t.c, t_version.c"


def test_collect_rejects_conflicting_declarations() -> None:
    metadata = MetaData()
    Table("t", metadata, Column("c", Enum("a", "b", name="shared")))
    Table("other", metadata, Column("c", Enum("a", "c", name="shared")))

    with pytest.raises(AmbiguousEnumDeclarationError, match="conflicting labels"):
        collect_declared_native_enums(metadata)


def test_add_value_revision(engine) -> None:
    """The end-to-end case: a label exists in the models and not in the database.

    Reaching the assertions at all proves the ops implement `reverse()`; autogenerate calls `reverse_into`
    unconditionally, so an op without it raises before a migration is ever written.
    """
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", "c", name="letters")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'c')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "add_value"},
        target_metadata=metadata,
        compare_enum_values=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_add_value.py").read_text()
    assert """op.execute("ALTER TYPE letters ADD VALUE IF NOT EXISTS 'b' BEFORE 'c'")""" in migration_contents

    run_alembic_command(engine=engine, command="upgrade", command_kwargs={"revision": "head"}, target_metadata=metadata)

    with engine.begin() as connection:
        labels = connection.execute(text("SELECT enum_range(NULL::letters)::text")).scalar()
    assert labels == "{a,b,c}"

    run_alembic_command(engine=engine, command="downgrade", command_kwargs={"revision": "base"}, target_metadata=metadata)


def test_no_revision_when_in_sync(engine) -> None:
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", name="letters")))

    with engine.begin() as connection:
        metadata.create_all(connection)

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "noop"},
        target_metadata=metadata,
        compare_enum_values=True,
    )

    assert "ALTER TYPE" not in (TEST_VERSIONS_ROOT / "1_noop.py").read_text()


def test_no_revision_when_opt_is_off(engine) -> None:
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", "c", name="letters")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'c')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "opt_off"},
        target_metadata=metadata,
    )

    assert "ALTER TYPE" not in (TEST_VERSIONS_ROOT / "1_opt_off.py").read_text()


def test_label_removed_from_models_raises(engine) -> None:
    """Postgres cannot drop an enum label, so this needs a human rather than a generated migration."""
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", name="letters")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'b')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    with pytest.raises(EnumLabelRemovedError, match=r"\['b'\]"):
        run_alembic_command(
            engine=engine,
            command="revision",
            command_kwargs={"autogenerate": True, "rev_id": "1", "message": "orphan"},
            target_metadata=metadata,
            compare_enum_values=True,
        )


def test_label_removed_from_models_is_tolerated_when_the_type_is_ignored(engine) -> None:
    """Dead labels cannot be dropped, so a consumer can vouch for a whole type instead."""
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "c", name="letters")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'b')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "ignored_orphan"},
        target_metadata=metadata,
        compare_enum_values=True,
        ignore_enum_label_removal={"letters"},
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_ignored_orphan.py").read_text()

    # The orphaned 'b' is tolerated, but a genuinely missing label is still added.
    assert """op.execute("ALTER TYPE letters ADD VALUE IF NOT EXISTS 'c' AFTER 'a'")""" in migration_contents


def test_ignoring_one_type_does_not_ignore_another(engine) -> None:
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", name="letters")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'b')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    with pytest.raises(EnumLabelRemovedError, match=r"\['b'\]"):
        run_alembic_command(
            engine=engine,
            command="revision",
            command_kwargs={"autogenerate": True, "rev_id": "1", "message": "other_orphan"},
            target_metadata=metadata,
            compare_enum_values=True,
            ignore_enum_label_removal={"some_other_type"},
        )


def test_creates_type_for_a_new_column_on_an_existing_table(engine) -> None:
    """`op.add_column` never emits CREATE TYPE, so without this the upgrade fails with `type does not exist`."""
    metadata = MetaData()
    Table("t", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", name="brand_new")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY)"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "new_type"},
        target_metadata=metadata,
        compare_enum_values=True,
        compare_tables=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_new_type.py").read_text()
    assert """op.execute("CREATE TYPE brand_new AS ENUM ('a', 'b')")""" in migration_contents

    run_alembic_command(engine=engine, command="upgrade", command_kwargs={"revision": "head"}, target_metadata=metadata)

    with engine.begin() as connection:
        assert connection.execute(text("SELECT enum_range(NULL::brand_new)::text")).scalar() == "{a,b}"


def test_does_not_create_type_when_the_table_is_also_new(engine) -> None:
    """`op.create_table` creates the type as a side effect, and CREATE TYPE has no IF NOT EXISTS to dedupe against."""
    metadata = MetaData()
    Table("brand_new_table", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", name="brand_new")))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "new_table"},
        target_metadata=metadata,
        compare_enum_values=True,
        compare_tables=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_new_table.py").read_text()
    assert "CREATE TYPE" not in migration_contents

    run_alembic_command(engine=engine, command="upgrade", command_kwargs={"revision": "head"}, target_metadata=metadata)

    with engine.begin() as connection:
        assert connection.execute(text("SELECT enum_range(NULL::brand_new)::text")).scalar() == "{a,b}"


def test_new_type_shared_by_new_and_existing_tables_raises(engine) -> None:
    metadata = MetaData()
    Table("existing", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", name="shared_new")))
    Table("brand_new_table", metadata, Column("id", Integer, primary_key=True), Column("c", Enum("a", "b", name="shared_new")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE existing (id integer PRIMARY KEY)"))

    with pytest.raises(MixedEnumTypeCreationError, match="Split this into separate migrations"):
        run_alembic_command(
            engine=engine,
            command="revision",
            command_kwargs={"autogenerate": True, "rev_id": "1", "message": "mixed"},
            target_metadata=metadata,
            compare_enum_values=True,
            compare_tables=True,
        )


def test_enum_ops_are_ordered_before_table_ops(engine) -> None:
    """A new label must exist before any statement that could reference it."""
    metadata = MetaData()
    Table(
        "t",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("c", Enum("a", "b", "c", name="letters")),
        Column("added_later", Integer),
    )

    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE letters AS ENUM ('a', 'c')"))
        connection.execute(text("CREATE TABLE t (id integer PRIMARY KEY, c letters)"))

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "1", "message": "ordering"},
        target_metadata=metadata,
        compare_enum_values=True,
        compare_tables=True,
    )

    migration_contents = (TEST_VERSIONS_ROOT / "1_ordering.py").read_text()

    assert migration_contents.index("ADD VALUE") < migration_contents.index("add_column")


def test_create_type_op_reverses_to_a_drop(engine) -> None:
    op = CreateEnumTypeOp("t", ("a", "b"), schema="s")

    assert op.reverse().sqltext == "DROP TYPE s.t"
    assert op.reverse().reverse().sqltext == op.sqltext
