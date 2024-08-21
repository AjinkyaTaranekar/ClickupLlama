from clickup_llama import ClickUpLlama

clickup_url = input("🐙 Enter ClickUp Docs URL: ")

try:
    workspace, doc_ids = clickup_url.split("app.clickup.com")[1].split("/v/dc/")
    doc_id, sub_doc_id = doc_ids.split("/")
except ValueError:
    print("❌ Invalid ClickUp URL. Please enter a valid URL.")
    exit(1)

print("⏳ Starting initiation of Ninja ⏳")
clickup_llama = ClickUpLlama(workspace, doc_id, sub_doc_id)

print(f"🦙 Clickup Llama initiated for repo: {workspace}/{doc_ids}")

while True:
    print(
        "-------------------------------------------------------------------------------"
    )
    query = input("🤔 Enter your query: ")

    try:
        answer = clickup_llama.answer_query(query)
        responses = answer.get("responses", "No response available.")
        sources = answer.get("sources", "No sources available.")
        print(f"\n\n📝 Answer to query: \n{responses} \n\nℹ️ Sources: {sources}")
    except Exception as e:
        print(f"❌ An error occurred while processing your query: {e}")

    print(
        "-------------------------------------------------------------------------------"
    )

    should_continue = input("Do you want to continue? (y/n): ").strip().lower()
    if should_continue == "n":
        break

print("👋 Exiting Clickup Llama. Goodbye!")
