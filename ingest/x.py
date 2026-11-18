from google.transit import gtfs_realtime_pb2

feed = gtfs_realtime_pb2.FeedMessage()

with open("C:\Real-Time Transport Monitoring\VehiclePositions", "rb") as f:
    feed.ParseFromString(f.read())

print("Number of entities:", len(feed.entity))

# Print the first vehicle entity (if any)
for entity in feed.entity:
    if entity.HasField("vehicle"):
        print(entity.vehicle)
        break
