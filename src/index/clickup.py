import os

import requests

CLICKUP_TOKEN = os.environ.get("CLICKUP_TOKEN")


def parse_response(response):
    if not isinstance(response, (list, dict)):
        raise ValueError("The response is not a list or dict.")

    parsed_data = []

    def extract_info(item):
        data = {
            "id": item.get("id"),
            "doc_id": item.get("doc_id"),
            "workspace_id": item.get("workspace_id"),
            "name": item.get("name"),
            "content": item.get("content"),
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
