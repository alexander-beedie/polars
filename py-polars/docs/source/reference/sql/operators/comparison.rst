Comparison
==========

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`>, >= <op_greater_than>`
     - Greater than (or equal to).
   * - :ref:`\<, \<= <op_less_than>`
     - Less than (or equal to).
   * - :ref:`==, !=, \<> <op_equal_to>`
     - Equal to (or not equal to).
   * - :ref:`IS [NOT] <op_is>`
     - Check if a value is one of the special values NULL, TRUE, or FALSE.
   * - :ref:`IS [NOT] DISTINCT FROM, \<=> <op_is_distinct_from>`
     - Check if a value is distinct from another value; this is the NULL-safe equivalent of `==` (and `!=`).
   * - :ref:`[NOT] BETWEEN <op_between>`
     - Check if a value is between two other values.

.. _op_greater_than:

GREATER THAN
------------
Returns True if the first value is greater than the second value.

| `>`: Greater than.
| `>=`: Greater than or equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n > 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 5000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n >= 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 3000 │
    # │ 5000 │
    # └──────┘


.. _op_less_than:

LESS THAN
---------
Returns True if the first value is less than the second value.

| `<`: Less than.
| `<=`: Less than or equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n < 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n <= 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 3000 │
    # └──────┘


.. _op_equal_to:

EQUAL
-----
Returns True if the two values are considered equal.

| `=`: Equal to.
| `!=`, `<>`: Not equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n = 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 3000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n != 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 5000 │
    # └──────┘


.. _op_is:

IS
--
| Returns True if the first value is identical to the second value (typically one of NULL, TRUE, or FALSE).
| Unlike `==` (and `!=`), this operator will always return TRUE or FALSE, never NULL.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"lbl": ["aa", "bb", "cc"], "n": [1000, None, 5000]})
    df.sql("SELECT * FROM self WHERE n IS NULL")
    # shape: (1, 2)
    # ┌─────┬──────┐
    # │ lbl ┆ n    │
    # │ --- ┆ ---  │
    # │ str ┆ i64  │
    # ╞═════╪══════╡
    # │ bb  ┆ null │
    # └─────┴──────┘

    df.sql("SELECT * FROM self WHERE n IS NOT NULL")
    # shape: (2, 2)
    # ┌─────┬──────┐
    # │ lbl ┆ n    │
    # │ --- ┆ ---  │
    # │ str ┆ i64  │
    # ╞═════╪══════╡
    # │ aa  ┆ 1000 │
    # │ cc  ┆ 5000 │
    # └─────┴──────┘


.. _op_is_distinct_from:

IS DISTINCT FROM
----------------
| Compare two values; this operator assumes that NULL values are equal.
| `IS NOT DISTINCT FROM` can also be written using the `<=>` operator.
| Equivalent to `==` (or `!=`) for non-NULL values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n1": [2222, None, 8888], "n2": [4444, None, 8888]})
    df.sql("SELECT * FROM self WHERE n1 IS DISTINCT FROM n2")
    # shape: (1, 2)
    # ┌──────┬──────┐
    # │ n1   ┆ n2   │
    # │ ---  ┆ ---  │
    # │ i64  ┆ i64  │
    # ╞══════╪══════╡
    # │ 2222 ┆ 4444 │
    # └──────┴──────┘

    df.sql("SELECT * FROM self WHERE n1 IS NOT DISTINCT FROM n2")
    df.sql("SELECT * FROM self WHERE n1 <=> n2")
    # shape: (2, 2)
    # ┌──────┬──────┐
    # │ n1   ┆ n2   │
    # │ ---  ┆ ---  │
    # │ i64  ┆ i64  │
    # ╞══════╪══════╡
    # │ null ┆ null │
    # │ 8888 ┆ 8888 │
    # └──────┴──────┘


.. _op_between:

BETWEEN
-------
Returns True if the first value is between the second and third values (inclusive).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 2000, 3000, 4000]})
    df.sql("SELECT * FROM self WHERE n BETWEEN 2000 AND 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 2000 │
    # │ 3000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n NOT BETWEEN 2000 AND 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 4000 │
    # └──────┘
