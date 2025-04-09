String
======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`BIT_LENGTH <bit_length>`
     - Returns the length of the input string in bits.
   * - :ref:`CONCAT <concat>`
     - Returns all input expressions concatenated together as a string.
   * - :ref:`CONCAT_WS <concat_ws>`
     - Returns all input expressions concatenated together (and interleaved with a separator) as a string.
   * - :ref:`DATE <date>`
     - Converts a formatted date string to an actual Date value.
   * - :ref:`ENCODE <encode>`
     - Encodes a string expression as 'hex' or 'base64'.
   * - :ref:`ENDS_WITH <ends_with>`
     - Returns True if the value ends with the second argument.
   * - :ref:`INITCAP <initcap>`
     - Converts the first letter of each word to uppercase, and the rest to lowercase.
   * - :ref:`LEFT <left>`
     - Returns the first (leftmost) `n` characters.
   * - :ref:`LENGTH <length>`
     - Returns the character length of the string.
   * - :ref:`LOWER <lower>`
     - Returns a lowercased column.
   * - :ref:`LTRIM <ltrim>`
     - Strips whitespaces from the left.
   * - :ref:`NORMALIZE <normalize>`
     - Convert string to the specified Unicode normalization form (one of NFC, NFD, NFKC, NFKD).
   * - :ref:`OCTET_LENGTH <octet_length>`
     - Returns the length of a given string in bytes.
   * - :ref:`REGEXP_LIKE <regexp_like>`
     - Returns True if `pattern` matches the value (optional: `flags`).
   * - :ref:`REPLACE <replace>`
     - Replaces a given substring with another string.
   * - :ref:`REVERSE <reverse>`
     - Returns the reversed string.
   * - :ref:`RIGHT <right>`
     - Returns the last (rightmost) `n` characters.
   * - :ref:`RTRIM <rtrim>`
     - Strips whitespaces from the right.
   * - :ref:`SPLIT_PART <split_part>`
     - Splits a string by another substring/delimiter, returning the `n`-th part; note that `n` is 1-indexed.
   * - :ref:`STARTS_WITH <starts_with>`
     - Returns True if the value starts with the second argument.
   * - :ref:`STRING_TO_ARRAY <string_to_array>`
     - Splits a string by another substring/delimiter, returning an array of strings.
   * - :ref:`STRPOS <strpos>`
     - Returns the index of the given substring in the target string; note that the result is 1-indexed
       (returning 0 indicates that the given string was not found).
   * - :ref:`STRPTIME <strptime>`
     - Converts a string to a Datetime using a strftime-compatible formatting string.
   * - :ref:`SUBSTR <substr>`
     - Returns a slice of the string data in the range [start, start + length]; note that `start` is 1-indexed.
   * - :ref:`TIMESTAMP <timestamp>` (or `DATETIME`)
     - Converts a formatted timestamp/datetime string to an actual Datetime value.
   * - :ref:`UPPER <upper>`
     - Returns an uppercased column.

.. _bit_length:

BIT_LENGTH
----------
Returns the length of the input string in bits.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["a", "bb", "ccc", "dddd"]})
    df.sql("""
      SELECT foo, BIT_LENGTH(foo) AS n_bits FROM self
    """)
    # shape: (4, 2)
    # ┌──────┬────────┐
    # │ foo  ┆ n_bits │
    # │ ---  ┆ ---    │
    # │ str  ┆ u32    │
    # ╞══════╪════════╡
    # │ a    ┆ 8      │
    # │ bb   ┆ 16     │
    # │ ccc  ┆ 24     │
    # │ dddd ┆ 32     │
    # └──────┴────────┘

.. _concat:

CONCAT
------
Returns all input expressions concatenated together as a string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"],
      }
    )
    df.sql("""
      SELECT CONCAT(foo, bar) AS foobar FROM self
    """)
    # shape: (4, 1)
    # ┌────────┐
    # │ foobar │
    # │ ---    │
    # │ str    │
    # ╞════════╡
    # │ aazz   │
    # │ byy    │
    # │ cxx    │
    # │ ddww   │
    # └────────┘

.. _concat_ws:

CONCAT_WS
---------
Returns all input expressions concatenated together (and interleaved with a separator) as a string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"],
      }
    )
    df.sql("""
      SELECT CONCAT_WS(':', foo, bar) AS foobar FROM self
    """)
    # shape: (4, 1)
    # ┌────────┐
    # │ foobar │
    # │ ---    │
    # │ str    │
    # ╞════════╡
    # │ aa:zz  │
    # │ b:yy   │
    # │ c:xx   │
    # │ dd:ww  │
    # └────────┘

.. _date:

DATE
----
Converts a formatted string date to an actual Date type; ISO-8601 format is assumed
unless a strftime-compatible formatting string is provided as the second parameter.

.. tip::

  `DATE` is also supported as a typed literal (this form does not allow a format string).

  .. code-block:: sql

    SELECT DATE '2021-01-01' AS dt

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "s_dt1": ["1969-10-30", "2024-07-05", "2077-02-28"],
        "s_dt2": ["10 February 1920", "5 July 2077", "28 April 2000"],
      }
    )
    df.sql("""
      SELECT
        DATE(s_dt1) AS dt1,
        DATE(s_dt2, '%d %B %Y') AS dt2
      FROM self
    """)
    # shape: (3, 2)
    # ┌────────────┬────────────┐
    # │ dt1        ┆ dt2        │
    # │ ---        ┆ ---        │
    # │ date       ┆ date       │
    # ╞════════════╪════════════╡
    # │ 1969-10-30 ┆ 1920-02-10 │
    # │ 2024-07-05 ┆ 2077-07-05 │
    # │ 2077-02-28 ┆ 2000-04-28 │
    # └────────────┴────────────┘

.. _encode:

ENCODE
------
Encodes a string expression as 'hex' or 'base64'.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"s": ["Hello", "World"]})
    df.sql("""
      SELECT s,
        ENCODE(s, 'base64') AS "s:b64",
        ENCODE(s, 'hex') AS "s:hex",
      FROM self
    """)
    # shape: (2, 3)
    # ┌───────┬──────────┬────────────┐
    # │ s     ┆ s:b64    ┆ s:hex      │
    # │ ---   ┆ ---      ┆ ---        │
    # │ str   ┆ str      ┆ str        │
    # ╞═══════╪══════════╪════════════╡
    # │ Hello ┆ SGVsbG8= ┆ 48656c6c6f │
    # │ World ┆ V29ybGQ= ┆ 576f726c64 │
    # └───────┴──────────┴────────────┘

.. _ends_with:

ENDS_WITH
---------
Returns True if the value ends with the second argument.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["aa", "bb", "cc", "dd"],
        "bar": ["zz", "yy", "xx", "ww"],
      }
    )
    df.sql("""
      SELECT foo, ENDS_WITH(foo, 'a') AS ends_a FROM self
    """)
    # shape: (4, 2)
    # ┌─────┬────────┐
    # │ foo ┆ ends_a │
    # │ --- ┆ ---    │
    # │ str ┆ bool   │
    # ╞═════╪════════╡
    # │ aa  ┆ true   │
    # │ bb  ┆ false  │
    # │ cc  ┆ false  │
    # │ dd  ┆ false  │
    # └─────┴────────┘

.. _initcap:

INITCAP
-------
Converts the first letter of each word to uppercase, and the rest to lowercase.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": ["hello world", "HELLO", "wOrLd"]})
    df.sql("""
      SELECT bar, INITCAP(bar) AS baz FROM self
    """)
    # shape: (3, 2)
    # ┌─────────────┬─────────────┐
    # │ bar         ┆ baz         │
    # │ ---         ┆ ---         │
    # │ str         ┆ str         │
    # ╞═════════════╪═════════════╡
    # │ hello world ┆ Hello World │
    # │ HELLO       ┆ Hello       │
    # │ wOrLd       ┆ World       │
    # └─────────────┴─────────────┘

.. _left:

LEFT
----
Returns the first (leftmost) `n` characters.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["abcd", "efgh", "ijkl", "mnop"],
        "bar": ["zz", "yy", "xx", "ww"],
      }
    )
    df.sql("""
      SELECT
        LEFT(foo, 1) AS foo1,
        LEFT(bar, 2) AS bar2
      FROM self
    """)

    # shape: (4, 2)
    # ┌──────┬──────┐
    # │ foo1 ┆ bar2 │
    # │ ---  ┆ ---  │
    # │ str  ┆ str  │
    # ╞══════╪══════╡
    # │ a    ┆ zz   │
    # │ e    ┆ yy   │
    # │ i    ┆ xx   │
    # │ m    ┆ ww   │
    # └──────┴──────┘

.. _length:

LENGTH
------
Returns the character length of the string.

.. admonition:: Aliases

   `CHAR_LENGTH`, `CHARACTER_LENGTH`

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "iso_lang":["de", "ru", "es"],
        "color": ["weiß", "синий", "amarillo"],
      }
    )
    df.sql("""
      SELECT
        iso_lang,
        color,
        LENGTH(color) AS n_chars,
        OCTET_LENGTH(color) AS n_bytes
      FROM self
    """)

    # shape: (3, 4)
    # ┌──────────┬──────────┬─────────┬─────────┐
    # │ iso_lang ┆ color    ┆ n_chars ┆ n_bytes │
    # │ ---      ┆ ---      ┆ ---     ┆ ---     │
    # │ str      ┆ str      ┆ u32     ┆ u32     │
    # ╞══════════╪══════════╪═════════╪═════════╡
    # │ de       ┆ weiß     ┆ 4       ┆ 5       │
    # │ ru       ┆ синий    ┆ 5       ┆ 10      │
    # │ es       ┆ amarillo ┆ 8       ┆ 8       │
    # └──────────┴──────────┴─────────┴─────────┘

.. _lower:

LOWER
-----
Returns a lowercased column.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["AA", "BB", "CC", "DD"]})
    df.sql("""
      SELECT foo, LOWER(foo) AS foo_lower FROM self
    """)
    # shape: (4, 2)
    # ┌─────┬───────────┐
    # │ foo ┆ foo_lower │
    # │ --- ┆ ---       │
    # │ str ┆ str       │
    # ╞═════╪═══════════╡
    # │ AA  ┆ aa        │
    # │ BB  ┆ bb        │
    # │ CC  ┆ cc        │
    # │ DD  ┆ dd        │
    # └─────┴───────────┘

.. _ltrim:

LTRIM
-----
Strips whitespaces from the left.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["   AA", " BB", "CC", "  DD"]})
    df.sql("""
      SELECT foo, LTRIM(foo) AS trimmed FROM self
    """)
    # shape: (4, 2)
    # ┌───────┬─────────┐
    # │ foo   ┆ trimmed │
    # │ ---   ┆ ---     │
    # │ str   ┆ str     │
    # ╞═══════╪═════════╡
    # │    AA ┆ AA      │
    # │  BB   ┆ BB      │
    # │ CC    ┆ CC      │
    # │   DD  ┆ DD      │
    # └───────┴─────────┘

.. _normalize:

NORMALIZE
---------
Convert string to the specified Unicode normalization form (one of NFC, NFD, NFKC, NFKD).
If the normalization form is not provided, NFC is used by default.

**Example:**

.. code-block:: python

    df = pl.DataFrame({
        "txt": [
            "Ｔｅｓｔ",
            "Ⓣⓔⓢⓣ",
            "𝕿𝖊𝖘𝖙",
            "𝕋𝕖𝕤𝕥",
            "𝗧𝗲𝘀𝘁",
        ],
    })
    df.sql("""
      SELECT NORMALIZE(txt, NFKC) FROM self
    """).to_series()
    # shape: (5,)
    # Series: 'txt' [str]
    # [
    #   "Test"
    #   "Test"
    #   "Test"
    #   "Test"
    #   "Test"
    # ]

.. _octet_length:

OCTET_LENGTH
------------
Returns the length of a given string in bytes.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "iso_lang":["de", "ru", "es"],
        "color": ["weiß", "синий", "amarillo"],
      }
    )
    df.sql("""
      SELECT
        iso_lang,
        color,
        OCTET_LENGTH(color) AS n_bytes,
        LENGTH(color) AS n_chars
      FROM self
    """)
    # shape: (3, 4)
    # ┌──────────┬──────────┬─────────┬─────────┐
    # │ iso_lang ┆ color    ┆ n_bytes ┆ n_chars │
    # │ ---      ┆ ---      ┆ ---     ┆ ---     │
    # │ str      ┆ str      ┆ u32     ┆ u32     │
    # ╞══════════╪══════════╪═════════╪═════════╡
    # │ de       ┆ weiß     ┆ 5       ┆ 4       │
    # │ ru       ┆ синий    ┆ 10      ┆ 5       │
    # │ es       ┆ amarillo ┆ 8       ┆ 8       │
    # └──────────┴──────────┴─────────┴─────────┘

.. _regexp_like:

REGEXP_LIKE
-----------
Returns True if `pattern` matches the value (optional: `flags`).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["abc123", "4ab4a", "abc456", "321cba"]})
    df.sql(r"""
      SELECT foo, REGEXP_LIKE(foo, '\d$') AS ends_in_digit FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬───────────────┐
    # │ foo    ┆ ends_in_digit │
    # │ ---    ┆ ---           │
    # │ str    ┆ bool          │
    # ╞════════╪═══════════════╡
    # │ abc123 ┆ true          │
    # │ 4ab4a  ┆ false         │
    # │ abc456 ┆ true          │
    # │ 321cba ┆ false         │
    # └────────┴───────────────┘

.. _replace:

REPLACE
-------
Replaces a given substring with another string.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["abc123", "11aabb", "bcbc45"]})
    df.sql("""
      SELECT foo, REPLACE(foo, 'b', '?') AS bar FROM self
    """)
    # shape: (3, 2)
    # ┌────────┬────────┐
    # │ foo    ┆ bar    │
    # │ ---    ┆ ---    │
    # │ str    ┆ str    │
    # ╞════════╪════════╡
    # │ abc123 ┆ a?c123 │
    # │ 11aabb ┆ 11aa?? │
    # │ bcbc45 ┆ ?c?c45 │
    # └────────┴────────┘

.. _reverse:

REVERSE
-------
Returns the reversed string.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["apple", "banana", "orange", "grape"]})
    df.sql("""
      SELECT foo, REVERSE(foo) AS oof FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬────────┐
    # │ foo    ┆ oof    │
    # │ ---    ┆ ---    │
    # │ str    ┆ str    │
    # ╞════════╪════════╡
    # │ apple  ┆ elppa  │
    # │ banana ┆ ananab │
    # │ orange ┆ egnaro │
    # │ grape  ┆ eparg  │
    # └────────┴────────┘

.. _right:

RIGHT
-----
Returns the last (rightmost) `n` characters.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["ab", "cde", "fghi", "jklmn"]})
    df.sql("""
      SELECT foo, RIGHT(foo, 2) AS bar FROM self
    """)
    # shape: (4, 2)
    # ┌───────┬─────┐
    # │ foo   ┆ bar │
    # │ ---   ┆ --- │
    # │ str   ┆ str │
    # ╞═══════╪═════╡
    # │ ab    ┆ ab  │
    # │ cde   ┆ de  │
    # │ fghi  ┆ hi  │
    # │ jklmn ┆ mn  │
    # └───────┴─────┘

.. _rtrim:

RTRIM
-----
Strips whitespaces from the right.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": ["zz    ", "yy  ", "xx ", "ww   "]})
    df.sql("""
      SELECT bar, RTRIM(bar) AS baz FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬─────┐
    # │ bar    ┆ baz │
    # │ ---    ┆ --- │
    # │ str    ┆ str │
    # ╞════════╪═════╡
    # │ zz     ┆ zz  │
    # │ yy     ┆ yy  │
    # │ xx     ┆ xx  │
    # │ ww     ┆ ww  │
    # └────────┴─────┘

.. _split_part:

SPLIT_PART
----------
Splits a string by another substring/delimiter, returning the `n`-th part; note that `n` is 1-indexed.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"s": ["xx,yy,zz", "abc,,xyz,???,hmm", None, ""]})
    df.sql("""
      SELECT
        s,
        SPLIT_PART(s,',',1) AS "s+1",
        SPLIT_PART(s,',',3) AS "s+3",
        SPLIT_PART(s,',',-2) AS "s-2",
      FROM self
    """)
    # shape: (4, 4)
    # ┌──────────────────┬──────┬──────┬──────┐
    # │ s                ┆ s+1  ┆ s+3  ┆ s-2  │
    # │ ---              ┆ ---  ┆ ---  ┆ ---  │
    # │ str              ┆ str  ┆ str  ┆ str  │
    # ╞══════════════════╪══════╪══════╪══════╡
    # │ xx,yy,zz         ┆ xx   ┆ zz   ┆ yy   │
    # │ abc,,xyz,???,hmm ┆ abc  ┆ xyz  ┆ ???  │
    # │ null             ┆ null ┆ null ┆ null │
    # │                  ┆      ┆      ┆      │
    # └──────────────────┴──────┴──────┴──────┘

.. _starts_with:

STARTS_WITH
-----------
Returns True if the value starts with the second argument.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["apple", "banana", "avocado", "grape"]})
    df.sql("""
      SELECT foo, STARTS_WITH(foo, 'a') AS starts_a FROM self
    """)
    # shape: (4, 2)
    # ┌─────────┬──────────┐
    # │ foo     ┆ starts_a │
    # │ ---     ┆ ---      │
    # │ str     ┆ bool     │
    # ╞═════════╪══════════╡
    # │ apple   ┆ true     │
    # │ banana  ┆ false    │
    # │ avocado ┆ true     │
    # │ grape   ┆ false    │
    # └─────────┴──────────┘

.. _string_to_array:

STRING_TO_ARRAY
---------------
Splits a string by another substring/delimiter, returning an array of strings.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["aa,bb,cc", "x,y"]})
    df.sql("""
      SELECT foo, STRING_TO_ARRAY(foo, ',') AS arr FROM self
    """)
    # shape: (2, 2)
    # ┌──────────┬────────────────────┐
    # │ foo      ┆ arr                │
    # │ ---      ┆ ---                │
    # │ str      ┆ list[str]          │
    # ╞══════════╪════════════════════╡
    # │ aa,bb,cc ┆ ["aa", "bb", "cc"] │
    # │ x,y      ┆ ["x", "y"]         │
    # └──────────┴────────────────────┘

.. _strpos:

STRPOS
------
Returns the index of the given substring in the target string.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["apple", "banana", "orange", "grape"]})
    df.sql("""
      SELECT foo, STRPOS(foo, 'a') AS pos_a FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬───────┐
    # │ foo    ┆ pos_a │
    # │ ---    ┆ ---   │
    # │ str    ┆ u32   │
    # ╞════════╪═══════╡
    # │ apple  ┆ 1     │
    # │ banana ┆ 2     │
    # │ orange ┆ 3     │
    # │ grape  ┆ 3     │
    # └────────┴───────┘


.. _strptime:

STRPTIME
--------
Converts a string to a Datetime using a `chrono strftime <https://docs.rs/chrono/latest/chrono/format/strftime/>`_-compatible formatting string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "s_dt": ["1969 Oct 30", "2024 Jul 05", "2077 Feb 28"],
        "s_tm": ["00.30.55", "12.40.15", "10.45.00"],
      }
    )
    df.sql("""
      SELECT
        s_dt,
        s_tm,
        STRPTIME(s_dt || ' ' || s_tm, '%Y %b %d %H.%M.%S') AS dtm
      FROM self
    """)
    # shape: (3, 3)
    # ┌─────────────┬──────────┬─────────────────────┐
    # │ s_dt        ┆ s_tm     ┆ dtm                 │
    # │ ---         ┆ ---      ┆ ---                 │
    # │ str         ┆ str      ┆ datetime[μs]        │
    # ╞═════════════╪══════════╪═════════════════════╡
    # │ 1969 Oct 30 ┆ 00.30.55 ┆ 1969-10-30 00:30:55 │
    # │ 2024 Jul 05 ┆ 12.40.15 ┆ 2024-07-05 12:40:15 │
    # │ 2077 Feb 28 ┆ 10.45.00 ┆ 2077-02-28 10:45:00 │
    # └─────────────┴──────────┴─────────────────────┘

.. _substr:

SUBSTR
---------
Returns a slice of the string data in the range [start, start + length]; note that `start` is 1-indexed.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["apple", "banana", "orange", "grape"]})
    df.sql("""
      SELECT foo, SUBSTR(foo, 3, 4) AS foo_3_4 FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬─────────┐
    # │ foo    ┆ foo_3_4 │
    # │ ---    ┆ ---     │
    # │ str    ┆ str     │
    # ╞════════╪═════════╡
    # │ apple  ┆ ple     │
    # │ banana ┆ nana    │
    # │ orange ┆ ange    │
    # │ grape  ┆ ape     │
    # └────────┴─────────┘


.. _timestamp:

TIMESTAMP
---------
Converts a formatted string date to an actual Datetime type; ISO-8601 format is assumed
unless a strftime-compatible formatting string is provided as the second parameter.

.. admonition:: Aliases

   `DATETIME`

.. tip::

  `TIMESTAMP` is also supported as a typed literal (this form does not allow a format string).

  .. code-block:: sql

    SELECT TIMESTAMP '2077-12-10 22:30:45' AS ts

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "str_timestamp": [
          "1969 July 30, 00:30:55",
          "2030-October-08, 12:40:15",
          "2077 February 28, 10:45:00",
        ]
      }
    )
    df.sql("""
      SELECT str_timestamp, TIMESTAMP(str_date, '%Y.%m.%d') AS date FROM self
    """)
    # shape: (3, 2)
    # ┌────────────┬────────────┐
    # │ str_date   ┆ date       │
    # │ ---        ┆ ---        │
    # │ str        ┆ date       │
    # ╞════════════╪════════════╡
    # │ 1969.10.30 ┆ 1969-10-30 │
    # │ 2024.07.05 ┆ 2024-07-05 │
    # │ 2077.02.28 ┆ 2077-02-28 │
    # └────────────┴────────────┘


.. _upper:

UPPER
-----
Returns an uppercased column.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["apple", "banana", "orange", "grape"]})
    df.sql("""
      SELECT foo, UPPER(foo) AS foo_upper FROM self
    """)
    # shape: (4, 2)
    # ┌────────┬───────────┐
    # │ foo    ┆ foo_upper │
    # │ ---    ┆ ---       │
    # │ str    ┆ str       │
    # ╞════════╪═══════════╡
    # │ apple  ┆ APPLE     │
    # │ banana ┆ BANANA    │
    # │ orange ┆ ORANGE    │
    # │ grape  ┆ GRAPE     │
    # └────────┴───────────┘
