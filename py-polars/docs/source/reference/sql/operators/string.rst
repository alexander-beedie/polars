
.. LIKE, "x ~~ y"
.. PNOT LIKE, "x !~~ y"
.. PILIKE,  "x ~~* y"
.. PNOT ILIKE,  "x !~~* y"
.. STARTS WITH,  ^@
.. CONCAT,  ||

String
======

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`[NOT] LIKE, ~~, !~~ <op_like>`
     - Matches a SQL "LIKE" expression.
   * - :ref:`[NOT] ILIKE, ~~*, !~~* <op_ilike>`
     - Matches a SQL "ILIKE" expression (case-insensitive "LIKE").

.. note::

    The `LIKE` operators match on string patterns, using the following wildcards:

    * The percent sign `%` represents zero or more characters.
    * An underscore `_` represents one character.


.. _op_like:

LIKE
----
| `~~`: Matches a SQL "LIKE" expression.
| `!~~`: Matches a SQL "ILIKE" expression (case-insensitive "LIKE").

**Example:**

.. code-block:: python



.. _op_ilike:

ILIKE
-----


.. _op_concat:

CONCAT
------


.. _op_starts_with:

STARTS_WITH
-----------
`^@`: Returns True if the first string value starts with the second string value.
