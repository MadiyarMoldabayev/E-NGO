import os
from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup

def html_table_to_markdown(html_content):
    """Converts a simple HTML table to a Markdown table."""
    soup = BeautifulSoup(html_content, 'lxml')
    table = soup.find('table')
    if not table:
        return ""

    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if cells:
            markdown += "| " + " | ".join(cells) + " |\n"
            
    return markdown

def process_pdf_to_markdown(pdf_path, output_path):
    """
    Extracts content from a PDF, preserving structure like tables, and saves it as Markdown.
    """
    print(f"Starting extraction from '{pdf_path}'...")
    
    # Using 'hi_res' strategy for better layout detection, especially with diagrams.
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True # Crucial for accurate table extraction
    )

    full_markdown_content = []
    
    print(f"Found {len(elements)} elements to process.")

    for el in elements:
        element_type = el.category
        
        if element_type == "Table":
            # Convert HTML table from element metadata to Markdown
            html_table = el.metadata.text_as_html
            if html_table:
                md_table = html_table_to_markdown(html_table)
                full_markdown_content.append(md_table)
        elif element_type == "ListItem":
            # Format list items with a bullet point
            full_markdown_content.append(f"- {el.text}")
        elif element_type in ["Title", "NarrativeText", "Uncategorized"]:
            # Append other text elements directly
            full_markdown_content.append(el.text)
        # We can add more specific handlers for other element types if needed

    # Join all markdown parts with appropriate spacing
    final_content = "\n\n".join(full_markdown_content)

    # Simple post-processing to clean up
    final_content = final_content.replace("-\n", "") # Fix hyphenated words
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    print(f"Extraction complete. Cleaned content saved to '{output_path}'.")


pdf_file = '251010-INPAS-Standard-Final.pdf'
output_file = '251010-INPAS-Standard-Final.md'

if os.path.exists(pdf_file):
    process_pdf_to_markdown(pdf_file, output_file)
else:
    print(f"Error: The file '{pdf_file}' was not found. Please check the path.")