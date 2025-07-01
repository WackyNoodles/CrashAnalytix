from pymongo import MongoClient
import os
from datetime import datetime
import sys
from bson import ObjectId

# Try local connection first, then Atlas if local fails
try:
    # Local MongoDB connection
    MONGO_URI = "mongodb://localhost:27017/"
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.server_info()
    print("Connected to local MongoDB")
except Exception as e:
    print(f"Local MongoDB connection failed: {e}")
    try:
        # MongoDB Atlas connection (update with your credentials)
        MONGO_URI = "mongodb+srv://251710311:tmS1ncvqydTAlyjo@cluster0.ubte6ty.mongodb.net/accident_detection?retryWrites=true&w=majority"
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        print("Connected to MongoDB Atlas")
    except Exception as e:
        print(f"MongoDB Atlas connection failed: {e}")
        print("ERROR: Could not connect to any MongoDB instance. Please check your configuration.")
        # Create a fallback in-memory storage for testing
        print("Using in-memory storage for testing (data will be lost when server restarts)")
        in_memory_accidents = []
        
# Database and collection names
db = client.accident_detection
accidents = db.accidents

def save_accident(accident_data):
    """Save accident data to MongoDB"""
    try:
        # Add timestamp if not present
        if "timestamp" not in accident_data:
            accident_data["timestamp"] = datetime.now().isoformat()
        
        # Insert the document
        result = accidents.insert_one(accident_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        # Fallback to in-memory storage if MongoDB fails
        if 'in_memory_accidents' in globals():
            in_memory_accidents.append(accident_data)
            return accident_data.get("_id", "unknown")
        return None

def get_all_accidents(sort_by_severity=False):
    """Get all accident records"""
    try:
        if sort_by_severity:
            # Custom sort order for severity: severe > moderate > minor
            pipeline = [
                {"$addFields": {
                    "severityOrder": {
                        "$switch": {
                            "branches": [
                                {"case": {"$eq": ["$severity", "Severe"]}, "then": 3},
                                {"case": {"$eq": ["$severity", "Moderate"]}, "then": 2},
                                {"case": {"$eq": ["$severity", "Minor"]}, "then": 1}
                            ],
                            "default": 0
                        }
                    }
                }},
                {"$sort": {"severityOrder": -1, "timestamp": -1}},
                {"$project": {"severityOrder": 0}}  # Remove the temporary field
            ]
            return list(accidents.aggregate(pipeline))
        else:
            # Default sort by timestamp (newest first)
            return list(accidents.find().sort("timestamp", -1))
    except Exception as e:
        print(f"Error fetching accidents: {e}")
        # Fallback to in-memory storage if MongoDB fails
        if 'in_memory_accidents' in globals():
            if sort_by_severity:
                # Sort by severity (simple implementation)
                severity_order = {"Severe": 3, "Moderate": 2, "Minor": 1}
                sorted_accidents = sorted(
                    in_memory_accidents, 
                    key=lambda x: (severity_order.get(x.get("severity", ""), 0), x.get("timestamp", "")),
                    reverse=True
                )
                return sorted_accidents
            else:
                # Sort by timestamp
                return sorted(in_memory_accidents, key=lambda x: x.get("timestamp", ""), reverse=True)
        return []

def get_accident_by_id(accident_id):
    """Get a single accident by ID"""
    try:
        return accidents.find_one({"_id": accident_id})
    except Exception as e:
        print(f"Error fetching accident by ID: {e}")
        # Fallback to in-memory storage if MongoDB fails
        if 'in_memory_accidents' in globals():
            for accident in in_memory_accidents:
                if accident.get("_id") == accident_id:
                    return accident
        return None