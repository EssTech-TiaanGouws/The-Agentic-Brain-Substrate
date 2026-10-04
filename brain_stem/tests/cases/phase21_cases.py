from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook
from src.swarm_core.structured_ingress import (
    StructuredFormat,
    StructuredIngressError,
    StructuredIngressLimits,
    StructuredIngressService,
)


def make_service(
    *,
    maximum_source_bytes: int = 1_000_000,
    maximum_rows: int = 10,
    maximum_columns: int = 8,
    maximum_cell_characters: int = 256,
) -> StructuredIngressService:
    return StructuredIngressService(
        StructuredIngressLimits(
            maximum_source_bytes=maximum_source_bytes,
            maximum_rows=maximum_rows,
            maximum_columns=maximum_columns,
            maximum_cell_characters=maximum_cell_characters,
        )
    )


def workbook_bytes(*, two_sheets: bool = False) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"
    worksheet.append(("name", "score"))
    worksheet.append(("Ada", 97))
    if two_sheets:
        workbook.create_sheet("Notes").append(("note",))
    output = io.BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


class StructuredIngressTests(unittest.TestCase):
    def test_csv_extracts_typed_rows_with_cell_coordinates_and_digest(self) -> None:
        content = b"name,score\nAda,97\nGrace,98\n"

        result = make_service().extract_bytes(
            content,
            format=StructuredFormat.CSV,
            source_ref="workspace:grades.csv",
        )

        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].fields[0].value, "Ada")
        self.assertEqual(result.records[0].fields[1].source_offset.coordinate, "B2")
        self.assertEqual(result.records[1].fields[0].source_offset.row_number, 3)
        self.assertEqual(len(result.source_sha256), 64)

    def test_jsonl_enforces_exact_fields_and_rejects_duplicate_keys(self) -> None:
        content = b'{"id":1,"value":"a"}\n{"value":"b","id":2}\n'
        result = make_service().extract_bytes(
            content,
            format=StructuredFormat.JSONL,
            source_ref="workspace:events.jsonl",
        )
        self.assertEqual(result.records[1].fields[0].value, 2)
        self.assertEqual(result.records[1].fields[1].value, "b")

        duplicate = b'{"id":1,"id":2}\n'
        with self.assertRaisesRegex(StructuredIngressError, "duplicate key"):
            make_service().extract_bytes(
                duplicate,
                format=StructuredFormat.JSONL,
                source_ref="workspace:duplicate.jsonl",
            )

    def test_xlsx_requires_sheet_selection_when_ambiguous_and_preserves_coordinates(self) -> None:
        content = workbook_bytes(two_sheets=True)
        service = make_service()

        with self.assertRaisesRegex(StructuredIngressError, "explicit worksheet"):
            service.extract_bytes(content, format=StructuredFormat.XLSX, source_ref="workbook:grades.xlsx")

        result = service.extract_bytes(
            content,
            format=StructuredFormat.XLSX,
            source_ref="workbook:grades.xlsx",
            sheet_name="Data",
        )
        self.assertEqual(result.records[0].fields[0].value, "Ada")
        self.assertEqual(result.records[0].fields[1].value, 97)
        self.assertEqual(result.records[0].fields[1].source_offset.coordinate, "B2")

    def test_sqlite_reads_only_requested_allowlisted_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "events.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute('CREATE TABLE "event data" ("user name" TEXT, amount INTEGER)')
            connection.execute('INSERT INTO "event data" VALUES (?, ?)', ("Ada", 3))
            connection.commit()
            connection.close()

            result = make_service().extract_sqlite_table(
                database_path,
                table_name="event data",
                columns=("user name", "amount"),
                source_ref="workspace:events.sqlite3",
            )
            self.assertEqual(tuple(field.value for field in result.records[0].fields), ("Ada", 3))
            self.assertEqual(result.records[0].fields[0].source_offset.sheet_name, "event data")

            with self.assertRaisesRegex(StructuredIngressError, "columns are not present"):
                make_service().extract_sqlite_table(
                    database_path,
                    table_name="event data",
                    columns=("amount; DROP TABLE event data",),
                    source_ref="workspace:events.sqlite3",
                )

            check = sqlite3.connect(database_path)
            self.assertEqual(check.execute('SELECT count(*) FROM "event data"').fetchone()[0], 1)
            check.close()

    def test_sqlite_reads_committed_wal_and_binds_selected_rows_to_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "events.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA wal_autocheckpoint=0")
            connection.execute("CREATE TABLE measurements (reading INTEGER)")
            connection.execute("INSERT INTO measurements VALUES (10)")
            connection.commit()

            first = make_service().extract_sqlite_table(
                database_path,
                table_name="measurements",
                columns=("reading",),
                source_ref="workspace:events.sqlite3",
            )
            connection.execute("INSERT INTO measurements VALUES (20)")
            connection.commit()
            second = make_service().extract_sqlite_table(
                database_path,
                table_name="measurements",
                columns=("reading",),
                source_ref="workspace:events.sqlite3",
            )
            connection.close()

        self.assertEqual(tuple(record.fields[0].value for record in first.records), (10,))
        self.assertEqual(tuple(record.fields[0].value for record in second.records), (10, 20))
        self.assertNotEqual(first.source_sha256, second.source_sha256)

    def test_byte_row_column_and_cell_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(StructuredIngressError, "byte limit"):
            make_service(maximum_source_bytes=4).extract_bytes(
                b"name\nAda\n",
                format=StructuredFormat.CSV,
                source_ref="large.csv",
            )
        with self.assertRaisesRegex(StructuredIngressError, "row limit"):
            make_service(maximum_rows=1).extract_bytes(
                b"name\nAda\nGrace\n",
                format=StructuredFormat.CSV,
                source_ref="rows.csv",
            )
        with self.assertRaisesRegex(StructuredIngressError, "column limit"):
            make_service(maximum_columns=1).extract_bytes(
                b"name,score\nAda,97\n",
                format=StructuredFormat.CSV,
                source_ref="columns.csv",
            )
        with self.assertRaisesRegex(StructuredIngressError, "character limit"):
            make_service(maximum_cell_characters=4).extract_bytes(
                b"name\nlong-value\n",
                format=StructuredFormat.CSV,
                source_ref="cell.csv",
            )

    def test_duplicate_headers_and_unapproved_format_are_rejected(self) -> None:
        with self.assertRaisesRegex(StructuredIngressError, "duplicate headers"):
            make_service().extract_bytes(
                b"name,name\nAda,Byron\n",
                format=StructuredFormat.CSV,
                source_ref="duplicate-headers.csv",
            )
        with self.assertRaises(StructuredIngressError):
            make_service().extract_bytes(
                b"unused",
                format="text/plain",
                source_ref="unapproved.txt",
            )


if __name__ == "__main__":
    unittest.main()