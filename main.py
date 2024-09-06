from src.graph.graph import app
from src.index.indexer import ingest_document

click_up_url = input("📝 Enter ClickUp Docs URL: ")
ingest_document(click_up_url)

while True:
    print(
        "-------------------------------------------------------------------------------"
    )
    query = input("🤔 User: Enter your query: ")
    if len(query.strip()):
        continue
    inputs = {"question": query}

    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            print("\n---\n")

        # Final generation
        print("🎯 LLM:", value["generation"])
    except Exception as e:
        print(f"❌ An error occurred while processing your query: {e}")

    print(
        "-------------------------------------------------------------------------------"
    )

    should_continue = input("Do you want to continue? (y/n): ").strip().lower()
    if should_continue == "n":
        break

print("👋 Exiting Clickup Llama. Goodbye!")
