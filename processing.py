import os
import re
from unstructured.partition.pdf import partition_pdf
from latex2mathml.converter import convert as convert_latex_to_mathml

# Step 1: Extract raw content from PDF
def extract_content_from_pdf(pdf_path, output_folder):
    return partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=output_folder
    )

# Step 2: Organize elements into defined categories
def classify_pdf_elements(elements):
    grouped = {
        "Header": [], "Footer": [], "Title": [], "NarrativeText": [],
        "Text": [], "ListItem": [], "Images": [], "Tables": []
    }

    for el in elements:
        el_type = type(el).__name__
        if el_type in grouped:
            grouped[el_type].append(str(el))
        elif el_type == "Image":
            grouped["Images"].append(str(el))
        elif el_type == "Table":
            grouped["Tables"].append(str(el))

    return grouped

# Step 3: Overall PDF processing pipeline
def process_pdf(file_path, base_temp_dir):
    image_output_dir = os.path.join(base_temp_dir, "pdf_extractions")
    os.makedirs(image_output_dir, exist_ok=True)

    raw_elements = extract_content_from_pdf(file_path, image_output_dir)
    categorized = classify_pdf_elements(raw_elements)

    # Add extracted image file paths
    categorized["Images"] = [
        os.path.join(image_output_dir, img)
        for img in os.listdir(image_output_dir)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    return categorized

# Optional utility: Format table strings into a grid
def parse_table_from_string(table_text):
    lines = table_text.strip().split('\n')
    rows = []

    for line in lines:
        # Normalize spacing and split into columns
        normalized = re.sub(r'\s+', ' ', line.strip())
        cells = normalized.split(' ')
        rows.append(cells)

    # Make all rows equally wide
    max_len = max(len(row) for row in rows)
    for row in rows:
        row.extend([''] * (max_len - len(row)))

    return rows

# Optional utility: Convert LaTeX inline expressions into MathML
def convert_latex_to_mathml_block(text):
    # Regex pattern for inline and block LaTeX
    pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'

    def render(match):
        latex_expr = match.group(1) or match.group(2)
        return convert_latex_to_mathml(latex_expr)

    return re.sub(pattern, render, text)
