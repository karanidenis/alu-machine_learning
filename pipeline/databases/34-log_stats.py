#!/usr/bin/env python3

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient()
db = client.logs
collection = db.nginx

# Count total number of documents
total_logs = collection.count_documents({})

print(f"{total_logs} logs")

# Count number of documents with each method
methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
for method in methods:
    count = collection.count_documents({"method": method})
    # print(f"    method {method}: {count}")
    print("method {}: {}".format(method, count))

# Count number of documents with method=GET and path=/status
status_check_count = collection.count_documents(
    {"method": "GET", "path": "/status"})
# print(f"{status_check_count} status check")
print("status check: {}".format(status_check_count))
