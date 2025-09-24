import pandas as pd
from openpyxl import load_workbook
import re


def extract_text_blocks(sheet, max_text_row=20):
    """
    Extract explanatory text blocks from the top or scattered areas.
    Scans rows until it finds a likely table header.
    """
    text_lines = []
    potential_header_keywords = [
        "NAME",
        "DESCRIPTION",
        "Attribute Series",
        "Industry name",
    ]  # Based on your screenshots

    for row_idx, row in enumerate(
        sheet.iter_rows(min_row=1, max_row=max_text_row), start=1
    ):
        row_values = [cell.value for cell in row if cell.value is not None]
        if not row_values:
            continue  # Skip empty rows

        # Check if this looks like a table header
        if any(
            keyword in str(val).upper()
            for val in row_values
            for keyword in potential_header_keywords
        ):
            break  # Stop at potential table start

        # Collect text (join cells in row)
        text_lines.append(" ".join(str(val) for val in row_values if val))

    explanations = "\n".join(line.strip() for line in text_lines if line.strip())
    return explanations, row_idx


def extract_table(sheet, start_row):
    """
    Extract the main table starting from a given row.
    Use pandas to read from that point.
    """
    from io import BytesIO

    temp_file = BytesIO()
    sheet._parent.save(temp_file)
    temp_file.seek(0)

    df = pd.read_excel(
        temp_file, sheet_name=sheet.title, header=start_row - 1, engine="openpyxl"
    )
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.applymap(lambda x: str(x).replace("\n", " ; ") if isinstance(x, str) else x)
    return df


def convert_excel_to_markdown(file_path, output_md="data_dict_markdown.md"):
    wb = load_workbook(file_path, data_only=True)
    markdown_docs = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]

        # Step 1: Extract text explanations and find table start
        explanations, table_start_row = extract_text_blocks(sheet)

        # Step 2: Extract table and convert to key-value format
        if table_start_row < sheet.max_row:
            df = extract_table(sheet, table_start_row)
            # Convert to key-value per row (screenshot format)
            table_kv = ""
            for _, row in df.iterrows():
                for col, value in row.items():
                    table_kv += f"{col}: {value}\n"
                table_kv += "\n"  # Blank line between records
        else:
            table_kv = "No main table detected."

        # Step 3: Combine into Markdown
        full_md = f"# Sheet: {sheet_name}\n## Explanations and Glossary:\n{explanations}\n\n## Main Table (Key-Value Format):\n{table_kv}"

        # Clean up extra newlines
        full_md = re.sub(r"\n{3,}", "\n\n", full_md)
        markdown_docs.append(full_md)

    # Save to file
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(markdown_docs))

    print(f"Conversion complete. Markdown saved to {output_md}")


# Usage
if __name__ == "__main__":
    convert_excel_to_markdown(
        "dummy_data_dictionary.xlsx"
    )  # Replace with your dummy_data_dictionary.xlsx or actual file
