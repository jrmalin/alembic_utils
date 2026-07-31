from alembic.operations import ops
from alembic.operations.ops import ExecuteSQLOp

from alembic_utils_extended.autogen_ordering import reorder_upgrade_ops
from alembic_utils_extended.pg_view import PGView
from alembic_utils_extended.reversible_op import CreateOp, DropOp, ReplaceOp

_VIEW = PGView(schema="public", signature="v", definition="SELECT 1 AS x")


def test_empty() -> None:
    upgrade_ops = ops.UpgradeOps(ops=[])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == []


def test_drops_before_stock() -> None:
    drop = DropOp(_VIEW)
    stock = ExecuteSQLOp("SELECT 1")
    upgrade_ops = ops.UpgradeOps(ops=[stock, drop])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [drop, stock]


def test_entity_creates_after_stock() -> None:
    create = CreateOp(_VIEW)
    stock = ExecuteSQLOp("SELECT 1")
    upgrade_ops = ops.UpgradeOps(ops=[create, stock])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [stock, create]


def test_index_creates_after_entity_creates() -> None:
    create = CreateOp(_VIEW)
    index = ops.CreateIndexOp("ix", "t", ["col"])
    upgrade_ops = ops.UpgradeOps(ops=[index, create])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [create, index]


def test_full_ordering() -> None:
    """Canonical upgrade order: library drops, stock ops, entity creates, index/constraint creates."""
    drop = DropOp(_VIEW)
    stock = ExecuteSQLOp("SELECT 1")
    create = CreateOp(_VIEW)
    index = ops.CreateIndexOp("ix", "t", ["col"])
    # Worst-case input order: index first, entity, stock, drop last.
    upgrade_ops = ops.UpgradeOps(ops=[index, create, stock, drop])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [drop, stock, create, index]


def test_replace_op_is_entity_create() -> None:
    replace = ReplaceOp(_VIEW)
    stock = ExecuteSQLOp("SELECT 1")
    upgrade_ops = ops.UpgradeOps(ops=[stock, replace])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [stock, replace]


def test_check_constraint_is_index_create() -> None:
    check = ops.CreateCheckConstraintOp("ck_foo", "t", "id > 0")
    create = CreateOp(_VIEW)
    upgrade_ops = ops.UpgradeOps(ops=[check, create])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [create, check]


def test_drop_index_is_library_drop() -> None:
    drop_idx = ops.DropIndexOp("ix", table_name="t")
    stock = ExecuteSQLOp("SELECT 1")
    upgrade_ops = ops.UpgradeOps(ops=[stock, drop_idx])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [drop_idx, stock]


def test_relative_order_preserved_within_groups() -> None:
    """Ops within the same group keep their original insertion order."""
    drop1 = DropOp(_VIEW)
    drop2 = ops.DropIndexOp("ix", table_name="t")
    stock1 = ExecuteSQLOp("SELECT 1")
    stock2 = ExecuteSQLOp("SELECT 2")
    create1 = CreateOp(_VIEW)
    create2 = ReplaceOp(_VIEW)
    index1 = ops.CreateIndexOp("ix1", "t", ["a"])
    index2 = ops.CreateCheckConstraintOp("ck", "t", "id > 0")

    upgrade_ops = ops.UpgradeOps(ops=[index1, stock1, create1, drop1, index2, stock2, create2, drop2])
    reorder_upgrade_ops(upgrade_ops)
    assert upgrade_ops.ops == [drop1, drop2, stock1, stock2, create1, create2, index1, index2]
