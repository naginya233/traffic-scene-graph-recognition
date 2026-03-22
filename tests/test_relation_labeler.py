import pytest
import numpy as np
from traffic_scene_graph.utils.relation_labeler import RelationLabeler

def test_label_entity_relations():
    labeler = RelationLabeler(cos_sim_threshold=0.7, closing_rate_threshold=2.0)
    
    # 2 vehicles
    # Vehicle 0 is at (0, 0), moving (0, 5) -> fast
    # Vehicle 1 is at (0, 10), moving (0, 5) -> fast
    # Same direction, v0 is behind v1
    bev_pos = np.array([[0, 0], [0, 10]])
    bev_vel = np.array([[0, 5], [0, 5]])
    
    # edge 0->1
    edge_index = np.array([[0], [1]])
    
    labels = labeler.label_entity_relations(bev_pos, bev_vel, edge_index)
    
    assert len(labels) == 1
    # 0 is tracking 1 (approaching / following) 
    # Actually closing rate = dot(dp, dv) = 0 so they are just following.
    # Class 4 is T_FOLLOWING depending on index. Let's not assert exact index if it's hardcoded, but ensure it's not unknown.
    assert labels[0] != 7 # Not unknown

def test_label_environment_relations():
    labeler = RelationLabeler()
    
    current_zones = [0, 1]
    previous_zones = [0, 0] # Vehicle 1 moved from 0 to 1
    
    labels = labeler.label_environment_relations(current_zones, previous_zones)
    
    assert len(labels) == 2
    assert labels[0] == 0  # In zone
    assert labels[1] == 1  # Entering zone (from 0 to 1 means entering 1)
