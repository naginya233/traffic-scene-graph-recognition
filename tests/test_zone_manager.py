import pytest
import numpy as np
import torch
from traffic_scene_graph.utils.zone_manager import ZoneManager

def test_zone_manager_init():
    zm = ZoneManager(bev_range=[0.0, 0.0, 60.0, 60.0], grid_rows=3, grid_cols=3, node_feature_dim=12)
    assert zm.num_zones == 9
    assert zm.zone_centers.shape == (9, 2)
    
    # Check if a center is correct. Range 0 to 60. Cells are 20x20. Center of first cell is (10, 10)
    assert np.allclose(zm.zone_centers[0], [10.0, 10.0])

def test_get_zone_index():
    zm = ZoneManager(bev_range=[0.0, 0.0, 60.0, 60.0], grid_rows=3, grid_cols=3)
    # 0: (0-20, 0-20), 1: (20-40, 0-20)...
    # 3: (0-20, 20-40)...
    pts = np.array([[10, 10], [50, 10], [10, 50], [50, 50]])
    idx = zm.get_zone_index(pts)
    assert len(idx) == 4
    assert idx[0] == 0  # row 0, col 0
    assert idx[1] == 2  # row 0, col 2
    assert idx[2] == 6  # row 2, col 0
    assert idx[3] == 8  # row 2, col 2

def test_generate_zone_features():
    zm = ZoneManager(bev_range=[0.0, 0.0, 60.0, 60.0], grid_rows=3, grid_cols=3, node_feature_dim=12)
    feat = zm.generate_zone_features()
    assert feat.shape == (9, 12)
    # Check is_zone flag
    assert torch.all(feat[:, -1] == 1.0)
    # Check coordinates in normalized space
    assert feat[0, 5] == 10.0 / 60.0
    assert feat[0, 6] == 10.0 / 60.0

def test_build_entity_zone_edges():
    zm = ZoneManager(bev_range=[0.0, 0.0, 60.0, 60.0], grid_rows=3, grid_cols=3)
    pts = np.array([[10, 10], [50, 50]])
    num_entity_nodes = 2
    zone_node_offset = 2
    
    edge_index, edge_attr = zm.build_entity_zone_edges(pts, num_entity_nodes, zone_node_offset)
    
    # 2 entities, each connects to its zone both ways -> 4 edges
    assert edge_index.shape == (2, 4)
    assert edge_attr.shape == (4, 4)
    
    # First entity connects to zone 0 (offset + 0 = 2)
    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 2
    
    assert edge_index[0, 1] == 2
    assert edge_index[1, 1] == 0
    
    # Second entity connects to zone 8 (offset + 8 = 10)
    assert edge_index[0, 2] == 1
    assert edge_index[1, 2] == 10
