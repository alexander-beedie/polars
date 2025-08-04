Logical
=======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Operator
     - Description
   * - :ref:`AND <op_logical_and>`
     - Combines conditions, returning True if both conditions are True.
   * - :ref:`OR <op_logical_or>`
     - Combines conditions, returning True if either condition is True.
   * - :ref:`NOT <op_logical_not>`
     - Negates a condition, returning True if the condition is False.


.. _op_logical_and:

AND
---
Combines conditions, returning True if both conditions are True.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE (foo >= bar) AND (bar < 50) AND (foo != 10)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 20  ┆ 10  │
    # │ 35  ┆ 35  │
    # └─────┴─────┘


.. _op_logical_or:

OR
--
Combines conditions, returning True if either condition is True.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE (foo % 2 != 0) OR (bar > 40)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 50  ┆ 50  │
    # │ 35  ┆ 35  │
    # └─────┴─────┘


.. _op_logical_not:

NOT
---
Negates a condition, returning True if the condition is False.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE NOT(foo % 2 != 0 OR bar > 40)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 10  ┆ 10  │
    # │ 20  ┆ 10  │
    # └─────┴─────┘
