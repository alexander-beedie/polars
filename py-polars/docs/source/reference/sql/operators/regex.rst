Regular Expression
==================

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`~, !~ <op_match_regex>`
     - Match a regular expression.
   * - :ref:`~*, !~* <op_match_regex_ci>`
     - Match a regular expression, case insensitively.


.. _op_match_regex:

REGEX MATCH
-----------
| `~`: Match a regular expression.
| `!~`: Does *not* match a regular expression.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"idx": [0, 1, 2, 3], "lbl": ["foo", "bar", "baz", "zap"]})
    df.sql("SELECT * FROM self WHERE lbl ~ '.*[aeiou]$'")
    # shape: (1, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ foo │
    # └─────┴─────┘

    df.sql("SELECT * FROM self WHERE lbl !~ '^b[aeiou]'")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ foo │
    # │ 3   ┆ zap │
    # └─────┴─────┘

.. _op_match_regex_ci:

REGEX MATCH (case insensitive)
------------------------------
| `~*`: Match a regular expression, case insensitively.
| `!~*`: Does *not* match a regular expression, case insensitively.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"idx": [0, 1, 2, 3], "lbl": ["FOO", "bar", "Baz", "ZAP"]})
    df.sql("SELECT * FROM self WHERE lbl ~* '.*[aeiou]$'")
    # shape: (1, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ FOO │
    # └─────┴─────┘

    df.sql("SELECT * FROM self WHERE lbl !~* '^b[aeiou]'")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ FOO │
    # │ 3   ┆ ZAP │
    # └─────┴─────┘
