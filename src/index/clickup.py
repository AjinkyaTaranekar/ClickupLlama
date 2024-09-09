import os
import re

import requests

CLICKUP_TOKEN = os.environ.get("CLICKUP_TOKEN")


def clean_markdown(text: str) -> str:
    # print("original:::", text)
    # Pattern for the complex nested structure
    pattern = r"\([^()]*(\([^()]*\)[^()]*)*\)"

    # Keep replacing the innermost parentheses until no more remain
    while re.search(pattern, text):
        text = re.sub(pattern, "", text)

    # Remove any remaining square brackets
    text = re.sub(r"\[.*?\]", "", text)

    # Remove any remaining parentheses
    text = re.sub(r"[\(\)]", "", text)

    # Convert Markdown tables to CSV-like format
    def table_to_csv(match):
        # Extract table rows
        rows = match.group().strip().split("\n")
        csv_lines = []
        for row in rows:
            # Split each row by "|" and strip whitespace
            csv_line = ", ".join(cell.strip() for cell in row.split("|")[1:-1])
            csv_lines.append(csv_line)
        return "\n".join(csv_lines)

    # Find Markdown tables and convert to CSV-like format
    text = re.sub(r"(\|.+?\|\n\|[-:| ]+\|\n(\|.+?\|\n)+)", table_to_csv, text)
    # Remove any occurrences of "---" followed by an optional comma
    text = re.sub(r"---[,]{0,1}", "", text)
    # Replace escaped underscores with regular underscores
    text = text.replace("\\_", "_")

    # print("transformed:::", text)
    return text.strip()


def parse_response(response):
    if not isinstance(response, (list, dict)):
        raise ValueError("The response is not a list or dict.")

    parsed_data = []

    def extract_info(item):
        cleaned_content = clean_markdown(item.get("content"))

        data = {
            "id": item.get("id"),
            "doc_id": item.get("doc_id"),
            "workspace_id": item.get("workspace_id"),
            "name": item.get("name"),
            "content": cleaned_content,
        }

        parsed_data.append(data)

        # Recursively extract pages if they exist
        if "pages" in item and isinstance(item["pages"], list):
            for page in item["pages"]:
                extract_info(page)

        return data

    if isinstance(response, list):
        for item in response:
            extract_info(item)
    elif isinstance(response, dict):
        extract_info(response)

    return parsed_data


def get_clickup_docs(workspace_id, doc_id, page_id=""):
    url = (
        f"https://api.clickup.com/api/v3/workspaces/{workspace_id}/docs/{doc_id}/pages"
    )

    # if page_id:
    #     url += "/" + page_id

    headers = {"Authorization": CLICKUP_TOKEN}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        documents = parse_response(data)
        print(f"📝 Got Documents from {len(documents)} ClickUp")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        documents = []
    except requests.exceptions.RequestException as req_err:
        print(f"Error during request: {req_err}")
        documents = []
    except ValueError as val_err:
        print(f"Error processing response: {val_err}")
        documents = []

    return documents
