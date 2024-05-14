from utils.config import boreas_dataroot, boreas_query_file, boreas_map_file
from utils.utils import *


if __name__ == "__main__":
    '''
    We use the boreas dataset as our test dataset.
    Set the dataroot to the boreas dataroot below.
    Use the json file created for a particular sequence as the query.
    Use a different sequence as the map.
    This script ensures that for each query in the query file, there 
    exists at least one sample in the map database with a threshold distance
    from the query.

    We use a distance threshold of 3m in our experiments.
    
    '''

    dataroot = boreas_dataroot
    query_file = boreas_query_file
    map_file = boreas_map_file

    map_poses = []
    query_poses = []

    # Use the correct sequences for query and map data
    with open(f"{dataroot}/{query_file}", 'r') as f:
        val_query = json.load(f)

    with open(f"{dataroot}/{map_file}", 'r') as f:
        val_map = json.load(f)

    # Extract query and map data    
    for key, value in val_query.items():
        query_poses.append(value['gps_pos'])

    for key, value in val_map.items():
        map_poses.append(value['gps_pos'])

    map_poses = np.array(map_poses)
    query_poses = np.array(query_poses)

    count = 0
    dist_threshold = 3.0

    map_indices = set()
    query_indices = []

    # At least 'min_req_samples' required
    min_req_samples = 1

    for query_idx, pose in enumerate(query_poses):
        # Compute distance of query sample from all map samples
        diff = np.linalg.norm(pose - map_poses, axis=1)

        # indices where the criteria is satisfied
        indices = np.where(diff < dist_threshold)[0]
        
        if len(indices) >= min_req_samples:
            # At least 'min_req_samples' available, use this query 
            # index and corresponding map indices
            query_indices.append(query_idx)
            for map_idx in indices:
                map_indices.add(map_idx)

            count += 1
    
    print(f"Total queries: {len(query_poses)}, Queries with map samples within {dist_threshold}m with more than {min_req_samples} samples: {count}")

    query_data = {}
    map_data = {}

    query_keys = list(val_query.keys())
    map_keys = list(val_map.keys())

    for query_idx in query_indices:
        query_data[query_keys[query_idx]] = val_query[query_keys[query_idx]]
    
    for map_idx in map_indices:
        map_data[map_keys[map_idx]] = val_map[map_keys[map_idx]]

    print(f"Number of query samples: {len(query_data.keys())}, Number of map samples: {len(map_data.keys())}")

    # Write the subset in new query and map files
    with open(f'{dataroot}/query_final.json', 'w') as f:
        json.dump(query_data, f)
    
    with open(f'{dataroot}/map_final.json', 'w') as f:
        json.dump(map_data, f)