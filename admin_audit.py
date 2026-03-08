from auth import _make_mongo_client
from tabulate import tabulate

def audit_learned_intelligence():
    client = _make_mongo_client()
    db = client["mental_health_db"]
    memory_collection = db["prediction_memory"]
    corrections_collection = db["model_corrections"]

    print("\n=== GLOBAL CONSENSUS KNOWLEDGE ===")
    global_knowledge = list(memory_collection.find({"learned_globally": True}))
    
    if not global_knowledge:
        print("No global knowledge learned yet.")
    else:
        table_data = []
        for entry in global_knowledge:
            table_data.append([
                entry.get("text"), 
                entry.get("label"), 
                entry.get("risk_level"),
                entry.get("last_updated").strftime("%Y-%m-%d %H:%M")
            ])
        print(tabulate(table_data, headers=["Phrase", "Corrected Label", "Risk", "Learned At"]))

    print("\n=== PENDING USER CORRECTIONS ===")
    pipeline = [
        {"$group": {
            "_id": {"text": "$text", "label": "$correct_label"},
            "count": {"$sum": 1}
        }},
        {"$match": {"count": {"$lt": 3}}}
    ]
    
    pending = list(corrections_collection.aggregate(pipeline))
    if not pending:
        print("No pending corrections.")
    else:
        pending_table = [[p["_id"]["text"], p["_id"]["label"], p["count"]] for p in pending]
        print(tabulate(pending_table, headers=["Phrase", "Suggested Label", "Votes"]))

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        from tabulate import tabulate
        
    audit_learned_intelligence()